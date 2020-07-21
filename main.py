# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:46:01 2019

@author: Tangmeii
"""
import numpy as np
import LCDA 

       
if __name__ == '__main__':
    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    k = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e+1))    
    basemodel = GaussianProcessRegressor(kernel=k,  
                                  alpha=0.0) 
    
    task = 'battery'
    if task == 'battery':    
        import scipy.io as sio
        data = sio.loadmat('Data/battery_feature_500.mat')
        Xs_ = data['B05_feature']
        Xt_ = data['B06_feature']
        Ys_ = data['B05_label'][0][:, np.newaxis]
        Yt_ = data['B06_label'][0][:, np.newaxis]
        
        X = np.concatenate((Xs_,Xt_))
        Y = np.concatenate((Ys_,Yt_))
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        pca_model = pca.fit(X)
        
        Xs_ = pca_model.transform(Xs_)
        Xt_ = pca_model.transform(Xt_)
        
        from sklearn.preprocessing import StandardScaler 
        scaler_x = StandardScaler() 
        scaler_x.fit(np.concatenate((Xs_, Xt_))) 
        Xs_ = scaler_x.transform(Xs_) 
        Xt_ = scaler_x.transform(Xt_)

    source_size = 0.5
    target_size = 0.05   
    n_rules = 6
    sigma    = 0.01
    wide_kernel = sigma*1
    lambda_inv  = 1
    Max_Iter    = 100 
    LR          = 1e-1
    lambda_regularization = 5e-1    
    
    from sklearn.model_selection import train_test_split
    Xs,_,Ys,_ = train_test_split(Xs_,Ys_,test_size= 1 - source_size, random_state = 10)    
    Xt,X_test,Yt,Y_test = train_test_split(Xt_,Yt_,test_size= 1 - target_size,random_state = 10)
    
    Y_pre = LCDA.model(Xs, Xt, Ys, Yt, X_test, n_rules, basemodel)
        
    from sklearn.metrics import mean_squared_error as mse
    er = mse(Y_test,Y_pre)
    print('Loss : ',er) 













