# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:34:40 2020

@author: Tangmeii
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.autograd import Variable


def logsumexp(x, dim, keepdim=False):
    """
    :param x:
    :param dim:
    :param keepdim:
    :return:
    """
    max, _ = torch.max(x, dim=dim, keepdim=True)
    out = max + (x - max).exp().sum(dim=dim, keepdim=keepdim).log()
    return out


def initialize(data, K, var = 0.1):
    '''
    Parameters
    ----------
    data: nb_samples * nb_features
    K : number of Gaussians
    param var: initial variance
    '''
    # Choose K points randomly as centers
    k_means = KMeans(init='k-means++', n_clusters=K, n_init=10) 
    k_means.fit(data)
    mu = k_means.cluster_centers_
    mu = torch.Tensor(mu)
      
    # Uniform sampling for means and variances
    d = data.size(1)
    var = torch.Tensor(K, d).fill_(var)
    
    # Uniform prior: latent variables z
    pi = torch.empty(K).fill_(1. /K)
    
    return mu, var, pi

def log_gaussian(x, mean=0, logvar=0.):
    '''
    Returns the density of x under the supplied gaussian
    Defaults to standard gaussian N(0, I)
    Parameters
    ----------
    x
    '''
    if type(logvar)=='float':
        logvar = x.new(1).fill_(logvar)
    a = (x - mean) ** 2
    log_p= -0.5*(np.log(2*np.pi) + logvar + a / logvar.exp())
    
    return log_p

def get_likelihoods(X, mu, logvar, log=True):
    '''
    Parameters
    ----------
    X: nb_samples, nb_features
    logvar : log-variances: K * features
    
    Returns
    -------
    likelihoods : K, nb_samples
    '''
    
    # Get feature-weise log-likelihood : K , nb_samples, nb_features
    log_likelihoods = log_gaussian(
            X[None, :, :], # (1, nb_samples, nb_features)
            mu[:, None, :], # (K, 1, nb_features)
            logvar[:, None, :], # (K, 1, nb_features)
            )
    
    # Sum over features
    log_likelihoods = log_likelihoods.sum(-1) # Notice sum not mean
    
    if not log:
        log_likelihoods.exp_() 
    
    return log_likelihoods

def get_density(mu, logvar, pi, N=50, X_range=(0, 5), Y_range=(0, 5)):
    """ Get the mesh to compute the density on. """
    X = np.linspace(*X_range, N)
    Y = np.linspace(*Y_range, N)
    X, Y = np.meshgrid(X, Y)
    
    # get the design matrix
    points = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    points = torch.from_numpy(points).float()
    
    # compute the densities under each mixture
    P = get_likelihoods(points, mu, logvar, log=False)
    
    # sum the densities to get mixture density
    Z = torch.sum(P, dim=0).data.numpy().reshape([N, N])
    
    return X, Y, Z    
    
def get_posteriors(log_likelihoods, log_pi):
    '''
    Calculate the posterior log p(z|x), assuming a uniform prior p(z)
    
    Parameters
    ----------
    likelihoods: the relative likelihoods p(x|z): (K, nb_samples)
    
    Return
    ------
    posteriors: (K, nb_samples)
    '''
#    posteriors = log_likelihoods - log_likelihoods.sum(0) # + log_pi[:, None]
    posteriors = log_likelihoods  + log_pi[:, None]
    posteriors = posteriors - logsumexp(posteriors, dim=0, keepdim=True)
    return posteriors
    
    
def get_parameters(X, log_posteriors, eps=1e-6, min_var=1e-6):
    ''' #PRML P439
    X: nb_samples * nb_features
    log_posteriors : p(z|x) K * nb_samples
    '''
    
    posteriors = log_posteriors.exp()
    
    # Compute 'N_k' the proxy 'number of points' assigned to each distribution
    K = posteriors.size(0)
    N_k = posteriors.sum(1)  # 
    N_k = N_k.view(K, 1, 1)
    
    # Get the means by taking the weighted combination of points
    # (K, 1, examples) @ (1, examples, features) -> (K, 1, features)
    mu = posteriors[:, None] @ X[None,]
    mu = mu / (N_k + eps) # PRML P439
    
    # Get the new var
    temp = X - mu
    var = (posteriors[:, None] @ (temp **2))/(N_k + eps) # (K, 1, features)
    logvar = torch.clamp(var, min=min_var).log()
    
    # Get the new mixing probabilities
    pi = N_k / N_k.sum()
    
    return mu.squeeze(1), logvar.squeeze(1), pi.squeeze()
    

def EM_fit(data, n_components, max_iter):
    # Train  
    data = torch.tensor(data)
    mu, var, pi = initialize(data, n_components, var=1)
    logvar = var.log()
    for i in range(max_iter):
        # get the likelihoods p(x|z) under the parameters
        log_likelihoods = get_likelihoods(data, mu, logvar, log=True)

        # Get Posteriors
        log_posteriors = get_posteriors(log_likelihoods, pi.log())
        
        # Updata Parameters
        mu, logvar, pi = get_parameters(data, log_posteriors, eps=1e-5, min_var=1e-5)
        
    return mu, logvar, pi    
 
    
def kernel(ker, X, X2, sigma):
    '''
    Pytorch
    Input: X  n_feature*Size1
           X2 n_feature*Size2
    Output: Size1*Size2
    '''
    n1, n2 = X.shape[1],X2.shape[1]
    n1sq = torch.sum(X ** 2, 0)        
    n2sq = torch.sum(X2 ** 2, 0)
    D = torch.ones((n1, n2), dtype = torch.double).mul(n2sq) +  \
        torch.ones((n2, n1), dtype = torch.double).mul(n1sq).t() - \
        2 * torch.mm(X.t(), X2)
    K = torch.exp(-sigma * D)
    return K
        
def model(X_source, X_target, Y_source, Y_target, X_test, n_rules, basemodel):
    
    n,  d   = X_source.shape  
    ns, nt = len(X_source), len(X_target)
    width = 10

    # Get mixture
    mu, logvar, pi = EM_fit(X_source, n_components=n_rules, max_iter=10)
    
    logvar = np.log(np.exp(logvar)*width)

    log_likelihoods = get_likelihoods(torch.tensor(X_target), mu, logvar, log=True)
    G = get_posteriors(log_likelihoods, pi.log()).t().detach().numpy()
    G = np.exp(G)
    
    log_likelihoods = get_likelihoods(torch.tensor(X_source), mu, logvar, log=True)
    Gs = get_posteriors(log_likelihoods, pi.log()).t().detach().numpy()
    Gs = np.exp(Gs)
    
    basemodel.fit(X_source, Y_source)
    Y_source = basemodel.predict(X_source).reshape(len(X_source), -1)
    E = Y_target - basemodel.predict(X_target).reshape(len(X_target), -1)
   
    
    # Initial parameters
    sigma    = 0.01
    wide_kernel = sigma*1
    lambda_inv  = 1
    Max_Iter    = 100 
    LR          = 1e-1
    lambda_regularization = 5e-1   

   
    # to torch 
    Xs = torch.from_numpy(X_source)
    Xt = torch.from_numpy(X_target)
    Ys = torch.from_numpy(Y_source)
    Yt = torch.from_numpy(Y_target)
    Gs_ = torch.from_numpy(Gs).double()
    Gt_ = torch.from_numpy(G).double()
    E_  = torch.from_numpy(E).double()
    
    # Kernel matrix [constant]
    KXs     = kernel('rbf', Xs.t(), Xs.t(), wide_kernel)
    KXt     = kernel('rbf', Xt.t(), Xt.t(), wide_kernel)
    KXs_inv = torch.inverse(KXs + lambda_inv*torch.eye(ns, dtype = torch.double))
    KXt_inv = torch.inverse(KXt + lambda_inv*torch.eye(nt, dtype = torch.double))    
    KXtXs   = kernel('rbf', Xt.t(), Xs.t(), wide_kernel)

    # initial params0 [constant]   
    params_W = torch.zeros((len(mu), 1)) 
    params_W = Variable(params_W, requires_grad=True)

    # Begin to optimize params
    Iteriation = 0
    
    # loss function 
    
    opt_SGD = torch.optim.Adam([params_W], lr=LR)
    while  (Iteriation < Max_Iter):
        Iteriation+=1
        W        = params_W               
        Ys_new   = Ys + Gs_.mm(W.double())
        temp     = Gt_.mm(W.double())
        tilde_K  = kernel('rbf', Ys_new.t(), Ys_new.t(), wide_kernel)
        tilde_Kc = kernel('rbf', Yt.t(), Ys_new.t(), wide_kernel)
        part1    = torch.trace(KXs_inv.mm(tilde_K).mm(KXs_inv).mm(KXs))
        part2    = 2 * torch.trace(KXs_inv.mm(tilde_Kc.t()).mm(KXt_inv).mm(KXtXs))
        part3    = lambda_regularization*\
                     (torch.sum(W.mul(W)))
        loss_cdm = part1 - part2 + part3
        opt_SGD.zero_grad()
        loss_cdm.backward()
        opt_SGD.step()    

    W = W.detach().numpy()
    W = W.reshape(len(W),1)
    R = Gs.dot(W)
    Ys_new = Y_source + R

    basemodel.fit(np.vstack((X_source, X_target)), np.vstack((Ys_new, Y_target)))             
    return basemodel.predict(X_test)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    