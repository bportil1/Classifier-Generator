import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)  
pd.set_option('display.width', 1000) 
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from os.path import dirname
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import inspection
from sklearn.manifold import TSNE
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.base import clone
import os
import time
from datetime import timedelta
from matplotlib.offsetbox import AnchoredText
import pickle
import scipy.spatial.distance
from scipy import spatial
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import *
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import discriminant_analysis
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings("ignore")

def generate_k_nearest_neighbours(X_train, y_train, X_test, y_test):
    metric = [ scipy.spatial.distance.mahalanobis,
               scipy.spatial.distance.canberra,
               scipy.spatial.distance.chebyshev, scipy.spatial.distance.correlation,
               scipy.spatial.distance.sqeuclidean, 
               'cityblock' , 'euclidean']
            
    fin_fut_objs = []
    exe = ThreadPoolExecutor(int(cpu_count()/4))
    for n in metric:
        fin_fut_objs.append(exe.submit(knn_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    
    return model1, best_rec

def knn_par_helper(metric_passed, X_train, y_train, X_test, y_test): 

    num_neigh = list(range(1,9))
    weights = ['uniform', 'distance', inverse_weights, squared_inverse]
    algorithm = ['ball_tree', 'kd_tree', 'brute']
    pred_best = 0
    model1 = ""
    best_rec = []
    for j in num_neigh:
        for m in weights:
            for n in algorithm:
                print(n, metric_passed, j, m)
                if metric_passed == scipy.spatial.distance.mahalanobis:
                    if n == 'kd_tree':
                        continue
                    model = sk.neighbors.KNeighborsClassifier(n_neighbors=j,
                                                              weights=m,
                                                              algorithm=n,
                                                              metric=metric_passed,
                                                              metric_params={'VI': np.linalg.inv(np.cov(X_train.T))},
                                                              n_jobs=1).fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    curr_score = sk.metrics.accuracy_score(y_test, y_pred)
                else:
                    if n == 'kd_tree' and metric_passed in [ scipy.spatial.distance.canberra, scipy.spatial.distance.chebyshev, scipy.spatial.distance.correlation, scipy.spatial.distance.sqeuclidean, ]:
                        continue
                    model = sk.neighbors.KNeighborsClassifier(n_neighbors=j,
                                                              weights=m,
                                                              algorithm=n,
                                                              metric=metric_passed,
                                                              n_jobs=1).fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    curr_score = sk.metrics.accuracy_score(y_test, y_pred)
                if curr_score > pred_best:
                    best_rec = (curr_score, n, metric_passed, j, m)
                    model1 = model
                    pred_best = curr_score 
                    print(model1, " ", pred_best, " ", best_rec)
    return model1, pred_best, best_rec
    
def inverse_weights(inpVec):
    return 1/inpVec

def squared_inverse(inpVec):
    return 1/(inpVec)**2

def generate_svc(X_train, y_train, X_test, y_test):

    kernel = [ 'rbf', 'sigmoid', 'poly', 'linear']
    fin_fut_objs = []
    exe = ThreadPoolExecutor(int(cpu_count()/8))
    
    for n in kernel:
        fin_fut_objs.append(exe.submit(svc_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
         
    #print(best_rec)
    return model1, best_rec

def svc_par_helper(passed_kernel, X_train, y_train, X_test, y_test):
    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0
    gamma = [ 'scale' , 'auto']
    C = [1 , .1, 10]
    coef0 = [0.0 , .5, 1.0]
    tol = [0.001 , .01, .0001]
    if passed_kernel == 'linear':
        for i in C:
            for j in tol:
                print((passed_kernel, i, j))
                model = sk.svm.SVC(C=i,
                                   kernel=passed_kernel,
                                   tol=j,
                                   random_state = 25,
                                   probability=True,
                                   max_iter = -1)
                model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                curr_score = model_scores.mean()
                model_preds.append((curr_score, passed_kernel, i, j))
                print((curr_score, passed_kernel, i, j))
                if curr_score > pred_best:
                    model1 = model
                    pred_best = curr_score
                    best_rec = (curr_score, passed_kernel, i, j) 
    elif passed_kernel == 'rbf':
        for i in C:
            for j in gamma:
                for k in tol:
                    print((passed_kernel, i, j, k))
                    model = sk.svm.SVC(C=i,
                                       kernel=passed_kernel,
                                       gamma=j,
                                       tol=k,
                                       random_state = 25,
                                       probability=True,
                                       max_iter = -1)
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                    curr_score = model_scores.mean()
                    model_preds.append((curr_score, passed_kernel, i, j, k))
                    print((curr_score, passed_kernel, i, j, k))
                    if curr_score > pred_best:
                        model1 = model
                        pred_best = curr_score
                        best_rec = (curr_score, passed_kernel, i, j, k) 
    elif passed_kernel == 'sigmoid':
        for i in C:
            for j in gamma:
                for k in coef0:
                    for l in tol:
                        print((passed_kernel, i, j, k, l))
                        model = sk.svm.SVC(C=i,
                                           kernel=passed_kernel,
                                           gamma=j,
                                           coef0=k,
                                           tol=l,
                                           random_state = 25,
                                           probability=True,
                                           max_iter = -1)
                        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                        curr_score = model_scores.mean()
                        model_preds.append((curr_score, passed_kernel, i, j, k, l))
                        print((curr_score, passed_kernel, i, j, k, l))
                        if curr_score > pred_best:
                            model1 = model
                            pred_best = curr_score
                            best_rec = (curr_score, passed_kernel, i, j, k, l) 
    elif passed_kernel == 'poly':
        degree = [1,2,3,5]
        for i in degree:
            for j in C:
                for k in gamma:
                    for l in coef0:
                        for m in tol:
                            print(passed_kernel, i, j, k, l, m)
                            model = sk.svm.SVC(C=j,
                                               kernel=passed_kernel,
                                               degree=i,
                                               gamma=k,
                                               coef0=l,
                                               tol=m,
                                               random_state = 25,
                                               probability=True,
                                               max_iter = -1)
                            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                            curr_score = model_scores.mean()
                            model_preds.append((curr_score, passed_kernel, j, i, k, l, m))
                            print((curr_score, passed_kernel, j, i, k, l, m))
                            if curr_score > pred_best:
                                model1 = model
                                pred_best = curr_score
                                best_rec = (curr_score, passed_kernel, j, i, k, l, m)

    return model1, pred_best, best_rec


def generate_gaussian_process(X_train, y_train, X_test, y_test):
            
    kernel = ['RBF', 'matern', 'rationalquadratic', 'dotproduct', 'linear', 'poly', 'laplacian', 'sigmoid', 'cosine' , 'None']  
    
    fin_fut_objs = []
    exe = ThreadPoolExecutor(int(cpu_count()/8))
    for n in kernel:
        fin_fut_objs.append(exe.submit(gp_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
        
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    print("At return line") 
    return model1, best_rec
    
def gp_par_helper(kernel_passed, X_train, y_train, X_test, y_test):
    model1 = ""
    best_rec = ()
    pred_best = 0
    curr_score = 0
    
    if kernel_passed == 'RBF':
        length_scale = [ 0.5, 1.1, 1.2, 1.3, 1.4, 1.5, 2.5 ]
        for i in length_scale:
            kernel_passed = sk.gaussian_process.kernels.RBF(length_scale = i)
            model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed, i)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score, i)
    elif kernel_passed == 'matern': 
        length_scale = [ 0.5, 1.1, 1.2, 1.3, 1.4, 1.5, 2.5 ]
        nu = [1.1,1.2,1.3,1.4,1.5]
        for i in length_scale:
            for j in nu:
                kernel_passed = sk.gaussian_process.kernels.Matern(length_scale = i, nu = j)
                model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                        optimizer='fmin_l_bfgs_b',
                                                                        n_restarts_optimizer=0,
                                                                        warm_start=True,
                                                                        max_iter_predict=100,
                                                                        random_state=25,
                                                                        n_jobs=1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed, i, j)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score, i, j)
                
    elif kernel_passed == 'rationalquadratic': 
        length_scale = [ 0.5, 1.1, 1.2, 1.3, 1.4, 1.5, 2.5 ]
        alpha = [5,5.1,5.2,5.3,5.4,5.5]
        for i in length_scale:
            for j in alpha:
                kernel_passed = sk.gaussian_process.kernels.RationalQuadratic(length_scale = i, alpha = j)
                model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                        optimizer='fmin_l_bfgs_b',
                                                                        n_restarts_optimizer=0,
                                                                        warm_start=True,
                                                                        max_iter_predict=100,
                                                                        random_state=25,
                                                                        n_jobs=1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed, i, j)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score, i, j)
    elif kernel_passed == 'None':
        model = sk.gaussian_process.GaussianProcessClassifier(  kernel = None,
                                                                optimizer='fmin_l_bfgs_b',
                                                                n_restarts_optimizer=0,
                                                                warm_start=True,
                                                                max_iter_predict=100,
                                                                random_state=25,
                                                                n_jobs=1) 
        print("Fitting model: ", kernel_passed)
        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
        curr_score = model_scores.mean()
        if curr_score > pred_best:
            best_rec = (curr_score, kernel_passed)
            model1 = model
            pred_best = curr_score 
        print(kernel_passed, curr_score)
    elif kernel_passed == 'dotproduct': 
        sigma = [2,4,6,8,10]
        for i in sigma:
            kernel_passed = sk.gaussian_process.kernels.DotProduct()
            model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score)
    elif kernel_passed == 'linear': 
        gamma = [.5, 1, 1.5]
        for i in gamma:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel( metric='linear')
            model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score)
    elif kernel_passed == 'poly': 
        gamma = [.1,.2,.3,.4,.5,.6,.7, 1.2, 1.3, 1.4, 1.5, 1.6]
        for i in gamma:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(gamma=i, metric='poly')
            model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed, i)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score, i)
    elif kernel_passed == 'laplacian': 
        gamma = [.5, 1, 1.5]
        for i in gamma:
            kernel_passed = sk.gaussian_process.kernels.PairwiseKernel(metric='laplacian')
            model = sk.gaussian_process.GaussianProcessClassifier(  kernel = kernel_passed,
                                                                    optimizer='fmin_l_bfgs_b',
                                                                    n_restarts_optimizer=0,
                                                                    warm_start=True,
                                                                    max_iter_predict=100,
                                                                    random_state=25,
                                                                    n_jobs=1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, kernel_passed)
                model1 = model
                pred_best = curr_score 
            print(kernel_passed, curr_score)   
    return model1, pred_best, best_rec

def generate_random_forest(X_train, y_train, X_test, y_test):    
    criterion = ['gini', 'entropy', 'log_loss']
    fin_fut_objs = []

    exe = ThreadPoolExecutor(int(cpu_count()/4))
    
    for n in criterion:
        fin_fut_objs.append(exe.submit(rf_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    
    return model1, best_rec

def rf_par_helper(criterion, X_train, y_train, X_test, y_test):
    n_est = [50,100,150]
    max_feature = ['sqrt', 'log2', None]
    min_samples_split = [2,3,5,6]
    min_samples_leaf = [1,2,3,4,5]
    pred_best = 0
    for i in n_est:
        for j in max_feature:
            for k in min_samples_split:
                for l in min_samples_leaf:
                    model = sk.ensemble.RandomForestClassifier(n_estimators=i,
                                                               criterion=criterion,
                                                               min_samples_split=k,
                                                               min_samples_leaf=l,
                                                               max_features=j,
                                                               n_jobs=1)
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                    curr_score = model_scores.mean()
                    if curr_score > pred_best:
                        best_rec = (curr_score, criterion, i, k, l, j)
                        model1 = model
                        pred_best = curr_score 
                    print(curr_score, criterion, i, k, l, j)
    return model1, pred_best, best_rec

def generate_hist_gradient_boosting_clf(X_train, y_train, X_test, y_test):    
    max_iter = [ 100, 125, 135]
    fin_fut_objs = []
    exe = ThreadPoolExecutor(int(cpu_count()/4))
    for n in max_iter:
        fin_fut_objs.append(exe.submit(hgbc_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    return model1, best_rec

def hgbc_par_helper(max_iter, X_train, y_train, X_test, y_test):
    pred_best = 0
    learning_rate = [.1, .2, .3]
    l2_regularization = [ 0, 1, 2]
    tol = [ 1*10**-9, 1*10**-7]
    
    for i in learning_rate:
        for j in l2_regularization:
            for k in tol:
                model = sk.ensemble.HistGradientBoostingClassifier(max_iter = max_iter,
                                                                   learning_rate = i,
                                                                   l2_regularization = j,
                                                                   tol = k,
                                                                   random_state = 42)
                model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                curr_score = model_scores.mean()
                if curr_score > pred_best:
                    best_rec = (curr_score, max_iter, i, j, k)
                    model1 = model
                    pred_best = curr_score 
                print(curr_score, max_iter, i, j, k)
    
    return model1, pred_best, best_rec

def generate_linear_svc(X_train, y_train, X_test, y_test):
    loss = ['hinge', 'squared_hinge']
    fin_fut_objs = []
    exe = ThreadPoolExecutor(int(cpu_count()/4))
    counter = 1
    for n in loss:
        print("Num of par jobs in inner parallelization: ", counter)
        counter = counter + 1
        fin_fut_objs.append(exe.submit(lsvc_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
 
    return model1, best_rec 

def lsvc_par_helper(loss, X_train, y_train, X_test, y_test):
    pred_best = 0
    penalty = ['l1', 'l2']
    tol = [ 1*10**-2, 1*10**-3, 1*10**-4, 1*10**-5 ]
    C = [.5, .7, 1, 1.3, 1.5]
    if loss == 'hinge':
        for i in tol:
            for j in C:
                model = sk.svm.LinearSVC(penalty='l2',
                                         loss=loss,
                                         dual=True,
                                         tol=i,
                                         C=j,
                                         random_state=42)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
            curr_score = model_scores.mean()
            if curr_score > pred_best:
                best_rec = (curr_score, loss, i, j)
                model1 = model
                pred_best = curr_score 
            print(curr_score, loss, 'l2', i, j)
                
    elif loss == 'squared_hinge':
        for k in penalty:
            for i in tol:
                for j in C:
                    dual_bool = True
                    if k == 'l1':
                        dual_bool = False
                    model = sk.svm.LinearSVC(penalty=k,
                                             loss=loss,
                                             dual=dual_bool,
                                             tol=i,
                                             C=j,
                                             random_state=42)
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                    curr_score = model_scores.mean()
                    if curr_score > pred_best:
                        best_rec = (curr_score, loss, i, j)
                        model1 = model
                        pred_best = curr_score 
                    print(curr_score, loss, k, i, j)
                
    return model1, pred_best, best_rec

def generate_adaboost_clf(X_train, y_train, X_test, y_test):    
    n_estimators = [ 50, 100, 125, 135]
    fin_fut_objs = []
    exe = ThreadPoolExecutor(int(cpu_count()/4))
    for n in n_estimators:
        fin_fut_objs.append(exe.submit(adaboost_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    return model1, best_rec

def adaboost_par_helper(n_estimators, X_train, y_train, X_test, y_test):
    pred_best = 0
    learning_rate = [.1, .2, .3, 1]
    for i in learning_rate:
        model = sk.ensemble.AdaBoostClassifier(n_estimators=n_estimators,
                                                           learning_rate = i,
                                                           random_state = 42)
        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
        curr_score = model_scores.mean()
        if curr_score > pred_best:
            best_rec = (curr_score, n_estimators, i)
            model1 = model
            pred_best = curr_score 
        print(curr_score, n_estimators, i) 
    return model1, pred_best, best_rec

def generate_qda(X_train, y_train, X_test, y_test):    
    reg_param = [0, .5, 1]
    fin_fut_objs = []

    exe = ThreadPoolExecutor(int(cpu_count()/4))
    
    for n in reg_param:
        fin_fut_objs.append(exe.submit(qda_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    
    return model1, best_rec
    
def qda_par_helper(reg_param, X_train, y_train, X_test, y_test):
    tol = [1*10**-2, 1*10**-3, 1*10**-4, 1*10**-5]
    pred_best = 0
    for i in tol:
        model = sk.discriminant_analysis.QuadraticDiscriminantAnalysis(reg_param=reg_param,
                                                                       tol=i)
        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
        curr_score = model_scores.mean()
        if curr_score > pred_best:
            best_rec = (curr_score, reg_param, i)
            model1 = model
            pred_best = curr_score 
            print(curr_score, reg_param, i)
    return model1, pred_best, best_rec

def generate_lda(X_train, y_train, X_test, y_test):    
    solver = ['svd', 'lsqr', 'eigen'] 
    fin_fut_objs = []

    exe = ThreadPoolExecutor(int(cpu_count()/4))
    
    for n in solver:
        fin_fut_objs.append(exe.submit(lda_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    
    return model1, best_rec
    
def lda_par_helper(solver, X_train, y_train, X_test, y_test):
    tol = [1*10**-2, 1*10**-3, 1*10**-4, 1*10**-5]
    pred_best = 0
    for i in tol:
        model = sk.discriminant_analysis.LinearDiscriminantAnalysis(solver=solver,
                                                                     tol=i)
        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
        curr_score = model_scores.mean()
        if curr_score > pred_best:
            best_rec = (curr_score, solver, i)
            model1 = model
            pred_best = curr_score 
            print(curr_score, solver, i)
    return model1, pred_best, best_rec

def generate_mlp(X_train, y_train, X_test, y_test):    
    activation = ['identity', 'logistic', 'tanh', 'relu'] 
    fin_fut_objs = []

    exe = ThreadPoolExecutor(int(cpu_count()/4))
    
    for n in activation:
        fin_fut_objs.append(exe.submit(mlp_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    
    return model1, best_rec
    
def mlp_par_helper(activation, X_train, y_train, X_test, y_test):
    hidden_layer_sizes = [100, 125]
    solver = ['adam' ] #, 'lbfgs', 'sgd']
    tol = [1*10**-3, 1*10**-4, 1*10**-5]
    pred_best = 0
    for i in solver:
        for j in hidden_layer_sizes:
            for k in tol:
                model = sk.neural_network.MLPClassifier(hidden_layer_sizes=j,
                                                        activation=activation,                                
                                                        solver=i,
                                                        tol=k)
                model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                curr_score = model_scores.mean()
                if curr_score > pred_best:
                    best_rec = (curr_score, activation, j, i, k)
                    model1 = model
                    pred_best = curr_score 
                    print(curr_score, activation, j, i, k)
    return model1, pred_best, best_rec

def generate_ridge(X_train, y_train, X_test, y_test):    
    solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']#, 'lbfgs'] 
    fin_fut_objs = []

    exe = ThreadPoolExecutor(int(cpu_count()/4))
    
    for n in solver:
        fin_fut_objs.append(exe.submit(ridge_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    
    return model1, best_rec
    
def ridge_par_helper(solver, X_train, y_train, X_test, y_test):
    alpha = [.5, 1, 1.5]
    fit_intercept = [True, False]
    tol = [1*10**-3, 1*10**-4, 1*10**-5]
    pred_best = 0
    for i in fit_intercept:
        for j in alpha:
            for k in tol:
                model = sk.linear_model.RidgeClassifier(alpha=j,
                                                        fit_intercept=i,                                
                                                        tol=k,
                                                        solver=solver)
                model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                curr_score = model_scores.mean()
                if curr_score > pred_best:
                    best_rec = (curr_score, solver, i, j, k)
                    model1 = model
                    pred_best = curr_score 
                    print(curr_score, solver, i, j, k)
    return model1, pred_best, best_rec

def generate_pa(X_train, y_train, X_test, y_test):

    loss = [ 'hinge', 'squared_hinge']
    fin_fut_objs = []
    exe = ThreadPoolExecutor(int(cpu_count()/8))
    
    for n in loss:
        fin_fut_objs.append(exe.submit(pa_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
         
    #print(best_rec)
    return model1, best_rec


def pa_par_helper(loss, X_train, y_train, X_test, y_test):
    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0

    C = [1 , .1, 10]
    tol = [0.001 , .01, .0001]
    for i in C:
        for j in tol:
            print((loss, i, j))
            model = sk.linear_model.PassiveAggressiveClassifier(C=i,
                                                               loss=loss,
                                                               tol=j,
                                                               n_jobs=1,
                                                               random_state = 25)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
            curr_score = model_scores.mean()
            model_preds.append((curr_score, loss, i, j))
            print((curr_score, loss, i, j))
            if curr_score > pred_best:
                model1 = model
                pred_best = curr_score
                best_rec = (curr_score, loss, i, j)
    return model1, pred_best, best_rec

def generate_sgd(X_train, y_train, X_test, y_test):

    loss = [ 'hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
    fin_fut_objs = []
    exe = ThreadPoolExecutor(int(cpu_count()/8))
    
    for n in loss:
        fin_fut_objs.append(exe.submit(sgd_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    return model1, best_rec

def sgd_par_helper(loss, X_train, y_train, X_test, y_test):
    model_preds = []
    best_rec = ()
    pred_best = 0
    curr_score = 0
    l1_ratio = .15
    penalty = ['l2', 'l1', 'elasticnet', None]
    alpha = [ .0001, 0.001]
    tol = [0.001 , .01, .0001]
    learning_rate = [ 'constant', 'optimal', 'invscaling', 'adaptive'] 
    for i in penalty:
        for j in learning_rate:
            if j in ['constant', 'invscaling', 'adaptive']:
                eta0 = .1
            else:
                eta0 = 0
            for k in alpha:
                for m in tol:
                    print((loss, i, j, k, m))
                    model = sk.linear_model.SGDClassifier(loss=loss,
                                                          penalty=i,
                                                          alpha=k,
                                                          tol=m,
                                                          n_jobs=1,
                                                          random_state = 25,
                                                          learning_rate=j,
                                                          eta0=eta0)
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                    curr_score = model_scores.mean()
                    model_preds.append((curr_score, loss, i, j, k, m))
                    print((curr_score, loss, i, j, k, m))
                    if curr_score > pred_best:
                        model1 = model
                        pred_best = curr_score
                        best_rec = (curr_score, loss, i, j, k, m)
    return model1, pred_best, best_rec

def generate_extra_trees_clf(X_train, y_train, X_test, y_test):    
    criterion = ['gini', 'entropy', 'log_loss']
    fin_fut_objs = []

    exe = ThreadPoolExecutor(int(cpu_count()/4))
    
    for n in criterion:
        fin_fut_objs.append(exe.submit(etc_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    
    return model1, best_rec

    
def etc_par_helper(criterion, X_train, y_train, X_test, y_test):
    n_est = [50,100,150]
    max_feature = ['sqrt', 'log2', None]
    min_samples_split = [2,3,5,6]
    min_samples_leaf = [1,2,3,4,5]
    pred_best = 0
    for i in n_est:
        for j in max_feature:
            for k in min_samples_split:
                for l in min_samples_leaf:
                    model = sk.ensemble.ExtraTreesClassifier(n_estimators=i,
                                                               criterion=criterion,
                                                               min_samples_split=k,
                                                               min_samples_leaf=l,
                                                               max_features=j,
                                                               n_jobs=1)
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
                    curr_score = model_scores.mean()
                    if curr_score > pred_best:
                        best_rec = (curr_score, criterion, i, k, l, j)
                        model1 = model
                        pred_best = curr_score 
                    print(curr_score, criterion, i, k, l, j)
    return model1, pred_best, best_rec

def generate_gnb(X_train, y_train, X_test, y_test):
    fin_fut_objs = []
    var_smoothing = [1*10**-7, 1*10**-8, 1*10**-9, 1*10**-10]
    exe = ThreadPoolExecutor(int(cpu_count()/4))
    for n in var_smoothing:
        fin_fut_objs.append(exe.submit(gnb_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    return model1, best_rec

def gnb_par_helper(var_smoothing, X_train, y_train, X_test, y_test):
    pred_best = 0
    model = sk.naive_bayes.GaussianNB(var_smoothing=var_smoothing)
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
    curr_score = model_scores.mean()
    if curr_score > pred_best:
        best_rec = (curr_score, var_smoothing)
        model1 = model
        pred_best = curr_score 
    print(curr_score, var_smoothing) 
    return model1, pred_best, best_rec

def generate_mnb(X_train, y_train, X_test, y_test):
    fin_fut_objs = []
    alpha = [0,.8, .9, 1, 1.1]
    exe = ThreadPoolExecutor(int(cpu_count()/4))
    for n in alpha:
        fin_fut_objs.append(exe.submit(mnb_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    return model1, best_rec

def mnb_par_helper(alpha, X_train, y_train, X_test, y_test):
    pred_best = 0
    force_alpha=False
    if alpha == 0:
        force_alpha=True
    model = sk.naive_bayes.MultinomialNB(alpha=alpha,
                                         force_alpha=force_alpha)
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
    curr_score = model_scores.mean()
    if curr_score > pred_best:
        best_rec = (curr_score, alpha, force_alpha) 
        model1 = model
        pred_best = curr_score 
    print(curr_score, alpha, force_alpha) 
    return model1, pred_best, best_rec

def generate_compnb(X_train, y_train, X_test, y_test):
    fin_fut_objs = []
    alpha = [0,.8, .9, 1, 1.1]
    exe = ThreadPoolExecutor(int(cpu_count()/4))
    for n in alpha:
        fin_fut_objs.append(exe.submit(compnb_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    return model1, best_rec

def compnb_par_helper(alpha, X_train, y_train, X_test, y_test):
    pred_best = 0
    force_alpha=False
    if alpha == 0:
        force_alpha=True
    model = sk.naive_bayes.ComplementNB(alpha=alpha,
                                         force_alpha=force_alpha)
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
    curr_score = model_scores.mean()
    if curr_score > pred_best:
        best_rec = (curr_score, alpha, force_alpha) 
        model1 = model
        pred_best = curr_score 
    print(curr_score, alpha, force_alpha) 
    return model1, pred_best, best_rec

def generate_bnb(X_train, y_train, X_test, y_test):
    fin_fut_objs = []
    alpha = [0,.8, .9, 1, 1.1]
    exe = ThreadPoolExecutor(int(cpu_count()/4))
    for n in alpha:
        fin_fut_objs.append(exe.submit(bnb_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    return model1, best_rec

def bnb_par_helper(alpha, X_train, y_train, X_test, y_test):
    pred_best = 0
    force_alpha=False
    if alpha == 0:
        force_alpha=True
    model = sk.naive_bayes.BernoulliNB(alpha=alpha,
                                       force_alpha=force_alpha)
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
    curr_score = model_scores.mean()
    if curr_score > pred_best:
        best_rec = (curr_score, alpha, force_alpha) 
        model1 = model
        pred_best = curr_score 
    print(curr_score, alpha, force_alpha) 
    return model1, pred_best, best_rec

def generate_isolation_forest(X_train, y_train, X_test, y_test):    
    n_estimators = [90, 100, 120]
    fin_fut_objs = []

    exe = ThreadPoolExecutor(int(cpu_count()/4))
    
    for n in criterion:
        fin_fut_objs.append(exe.submit(if_par_helper, n, X_train, y_train, X_test, y_test))
    
    for obj in range(len(fin_fut_objs)):
        fin_fut_objs[obj] = fin_fut_objs[obj].result()
    fin_fut_objs = np.asarray(fin_fut_objs, dtype='object')
    clf_acc_high_idx = np.argmax(fin_fut_objs[:,1])
    model1 = fin_fut_objs[clf_acc_high_idx][0]
    best_rec = fin_fut_objs[clf_acc_high_idx][2]
    
    return model1, best_rec

def if_par_helper(n_est, X_train, y_train, X_test, y_test):
    model = sk.ensemble.IsolationForestClassifier(n_estimators=n_est,
                                                  n_jobs=1)
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs=1 )
    curr_score = model_scores.mean()
    if curr_score > pred_best:
        best_rec = (curr_score, n_est)
        model1 = model
        pred_best = curr_score 
        print(curr_score, n_est)
    return model1, pred_best, best_rec

def supervised_learning_caller_optimized(alg, X_train, y_train, X_test, y_test):
    if alg == 'knn':
        start_time = time.monotonic()
        model, best_rec = generate_k_nearest_neighbours(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic()-start_time)
    elif alg == 'svc':
        start_time = time.monotonic()
        model, best_rec = generate_svc(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gp':
        start_time = time.monotonic()
        model, best_rec = generate_gaussian_process(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'rf':
        start_time = time.monotonic()
        model, best_rec = generate_random_forest(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'lsvc':
        start_time = time.monotonic()
        model, best_rec = generate_linear_svc(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'hgbc':
        start_time = time.monotonic()
        model, best_rec = generate_hist_gradient_boosting_clf(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'ada':
        start_time = time.monotonic()
        model, best_rec = generate_adaboost_clf(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'qda':
        start_time = time.monotonic()
        model, best_rec = generate_qda(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'lda':
        start_time = time.monotonic()
        model, best_rec = generate_qda(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'mlp':
        start_time = time.monotonic()
        model, best_rec = generate_mlp(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'ridge':
        start_time = time.monotonic()
        model, best_rec = generate_ridge(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'pa':
        start_time = time.monotonic()
        model, best_rec = generate_pa(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'sgd':
        start_time = time.monotonic()
        model, best_rec = generate_sgd(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'etc':
        start_time = time.monotonic()
        model, best_rec = generate_extra_trees_clf(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'gnb':
        start_time = time.monotonic()
        model, best_rec = generate_gnb(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'mnb':
        start_time = time.monotonic()
        model, best_rec = generate_mnb(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'compnb':
        start_time = time.monotonic()
        model, best_rec = generate_compnb(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'bnb':
        start_time = time.monotonic()
        model, best_rec = generate_bnb(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)
    elif alg == 'if':
        start_time = time.monotonic()
        model, best_rec = generate_isolation_forest(X_train, y_train, X_test, y_test)
        training_time = timedelta(seconds=time.monotonic() - start_time)  

            
    return model, best_rec, training_time

def plain_clf_runner(alg):
    ### Binary Classification Algorithms
    if alg == 'knn':
        model = sk.neighbors.KNeighborsClassifier()
    elif alg == 'lsvc':
        model = sk.svm.LinearSVC()
    elif alg == 'nsvc':
        model = sk.svm.NuSVC(probability=True)
    elif alg == 'svc':
        model = sk.svm.SVC(probability=True)
    elif alg == 'gp':
        model = sk.gaussian_process.GaussianProcessClassifier()
    elif alg == 'dt':
        model = sk.tree.DecisionTreeClassifier()
    elif alg == 'rf':
        model = sk.ensemble.RandomForestClassifier()
    elif alg == 'ada':
        model = sk.ensemble.AdaBoostClassifier()
    elif alg == 'gnb':
        model = sk.naive_bayes.GaussianNB()
    elif alg == 'mnb':
        model = sk.naive_bayes.MultinomialNB()
    elif alg == 'compnb':
        model = sk.naive_bayes.ComplementNB()
    elif alg == 'qda':
        model = sk.discriminant_analysis.QuadraticDiscriminantAnalysis()
    elif alg == 'lda':
        model = sk.discriminant_analysis.LinearDiscriminantAnalysis()
    elif alg == 'mlp':
        model = sk.neural_network.MLPClassifier()
    elif alg == 'ridge':
        model = sk.linear_model.RidgeClassifier()
    elif alg ==  'pa':
        model = sk.linear_model.PassiveAggressiveClassifier()
    elif alg ==  'sgd':
        model = sk.linear_model.SGDClassifier()
    elif alg == 'etc':
        model = sk.ensemble.ExtraTreesClassifier()
    elif alg == 'hgbc':
        model = sk.ensemble.HistGradientBoostingClassifier()
    ### Multiclass Only Classification Algorithms
    elif alg == 'bnb':
        model = sk.naive_bayes.BernoulliNB()
    elif alg == 'if':
        model = sk.ensemble.IsolationForest() 
    
    return model

def binarize_data(data, data_labels):
    idx = 0
    num_labels = []
    bin_data = data.copy(deep=True)
    for label in data_labels:
        num_labels.append(idx)
        bin_data['label'] = bin_data['label'].mask(bin_data['label']==label, idx)
        idx = idx + 1
    bin_data = bin_data.to_numpy()
    return bin_data, num_labels

def classifier_name_expand(alg):
    classifier_name_dict = {
        'knn':'K-Nearest Neighbors',
        'lsvc': 'Linear SVC',
        'svc': 'SVC',
        'gp': 'Gaussian Processes',
        'dt': 'Decision Tree',
        'rf': 'Random Forest',
        'ada': 'AdaBoost',
        'gnb': 'Gaussian Naive Bayes',
        'mnb': 'Multinomial Naive Bayes',
        'compnb': 'Complement Naive Bayes',
        'bnb': 'Bernoulli Naive Bayes',
        'qda': 'Quadratic Discriminant Analysis',
        'lda': 'Linear Discriminant Analysis',
        'mlp': 'Multilayer Perceptron',
        'ridge': 'Ridge',
        'pa': 'Passive Aggressive',
        'sgd': 'SGD',
        'hgbc': 'Histogram Gradient Boosting Classifier',
        'etc': 'Extra Trees Classifier',
        'if': 'Isolation Forest'
    }
    return classifier_name_dict[alg]

sem = True

def supervised_methods_evaluation(alg, model, X, y, X_test, y_test,
                                  X_train_2d, X_test_2d,
                                  X_train_size, y_train_size,
                                  output_path, data_labels,
                                  clf_type, class_type,
                                  optimizing_tuple=None, optimization_time=None):
    
    global sem

    model_2d = clone(model)
    model_2d = model_2d.fit(X_train_2d, y)
    y_bin, bin_labels = binarize_data(y_test, data_labels)
  
    start_time = time.monotonic()
    acc_scores = cross_val_score(model, X, y, cv = 5, n_jobs=1)
    cv_time = timedelta(seconds=time.monotonic() - start_time)
    
    start_time = time.monotonic()
    model.fit(X,y)
    fitting_time = timedelta(seconds=time.monotonic() - start_time)
    
    start_time = time.monotonic()
    y_pred = model.predict(X_test)
    prediction_time = timedelta(seconds=time.monotonic() - start_time)
    
    output_file_path = output_path + "/supervised_methods_evaluation_results/" + alg + "/"
    
    if clf_type == 'optimized':
        output_path = output_file_path + "optimized_results/"  
        summary_file_path = output_path + alg + "_summary_optimized.csv"
        output_file_path = output_path + alg +"_optimized_" + "results.txt"
        roc_data_path = output_path  + alg + "_roc_data.csv"
    else:
        output_path = output_file_path + "plain_results/"
        summary_file_path = output_path  + alg + "_summary_plain.csv"
        output_file_path = output_path  + alg + "_results.txt"
        roc_data_path = output_path  + alg + "_roc_data.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_file = open(output_file_path, "w")
    #output_file2 = open(summary_file_path, "a")
    output_file3 = open(roc_data_path, "w")
    
    output_file.write("Classifier Name: " + classifier_name_expand(alg))

    output_file.write("\nCV Time: " + str(cv_time) + ", ")

    output_file.write("\nFitting Time: " + str(fitting_time) + ", ")

    output_file.write("\nPrediction Time: " + str(prediction_time) + ", ")

    accScore = sk.metrics.accuracy_score(y_test, y_pred)
    output_file.write("\nAccuracy Score on Test Set: " + str(accScore))
        
    output_file.write("\nAccuracy Scores During Training: " + str(acc_scores))
    output_file.write("\nAccuracy Scores Mean During Training: " + str(acc_scores.mean()))
    output_file.write("\nAccuracy Scores Standard Deviation During Training: " + str(acc_scores.std()))

    preScore = sk.metrics.precision_score(y_test, y_pred, average="weighted", labels = data_labels)
    output_file.write("\nPrecision Score: " + str(preScore))
    
    recScore = sk.metrics.recall_score(y_test, y_pred, average="weighted", labels = data_labels)
    output_file.write("\nRecall Score: " + str(recScore))               

    f1Score = sk.metrics.f1_score(y_test, y_pred, average="weighted", labels = data_labels)
    output_file.write("\nF1 Score: " + str(f1Score))
                     
    f2Score = sk.metrics.fbeta_score(y_test, y_pred, beta = 1.2, average="weighted", labels = data_labels)
    output_file.write("\nF2 Score: " + str(f2Score))                     
    

    while sem == False:
        pass
    
    sem = False
    plt.clf()
    cm = sk.metrics.confusion_matrix(y_test, y_pred)
    output_file.write("\nConfusion Matrix: \n" + str(cm))
    cmdisp = sk.metrics.ConfusionMatrixDisplay(cm, display_labels = data_labels)
    cmdisp.plot()
    plt.title(classifier_name_expand(alg) + " Confusion Matrix")
    plt.savefig(output_path + alg + " cm.png")
    plt.close()
    ### binary case block
    
    if class_type == 'binary':
        if alg not in ['lsvc', 'ridge', 'pa', 'sgd', 'perc']:
            y_score = model.predict_proba(X_test)
            y_score_rav = y_score[:,1]
        else:
            y_score_rav = model.decision_function(X_test)
        plt.clf()
        ras = sk.metrics.roc_auc_score(y_test, y_score_rav, average='weighted')
        fpr, tpr, thrsh = sk.metrics.roc_curve(y_test, y_score_rav, pos_label='typed')
        plt.plot(fpr, tpr, label=classifier_name_expand(alg))
        plt.title(classifier_name_expand(alg) + " ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        text_box = AnchoredText("ROC AUC Score: " + str(ras), frameon=True, loc=4, pad=0.5)
        plt.setp(text_box.patch, facecolor='white', alpha=0.5)
        plt.gca().add_artist(text_box)
        plt.savefig(output_path + alg + "_ras.png")
        plt.close()

        output_file.write("\nROC AUC Score: " + str(ras))     
        
        data = pd.DataFrame({ 'fpr' : fpr , 'tpr' : tpr, 'thrsh' : thrsh  })
        data.to_csv(roc_data_path, index = False)

    ###
    plt.clf()
    color = np.array(np.random.choice(range(256), size=len(data_labels)))
    dbd = DecisionBoundaryDisplay.from_estimator(model_2d, X_test_2d, response_method="predict",  xlabel="X-values", ylabel="Y-values", alpha=.5)
    dbd.plot()
    plt.scatter(X_test_2d[:,0].astype(int), X_test_2d[:,1].astype(int), c=color[y_bin.astype(int)], edgecolor="k")
    plt.title(classifier_name_expand(alg) + " Decision Boundary Display")
    plt.savefig(output_path + alg + " dbd.png")
    plt.close()
    
    sem = True

    mcc = sk.metrics.matthews_corrcoef(y_test, y_pred)
    output_file.write("\nMatthews Correlation Coefficient: " + str(mcc))

    ck = sk.metrics.cohen_kappa_score(y_test, y_pred)
    output_file.write("\nCohen's kappa: " + str(ck))

    jaccard = sk.metrics.jaccard_score(y_test, y_pred, average="weighted", labels = data_labels)
    output_file.write("\nJaccard Score: " + str(jaccard))
    
    hammingl = sk.metrics.hamming_loss(y_test, y_pred)
    output_file.write("\nHamming Loss: " + str(hammingl))

    z1l = sk.metrics.zero_one_loss(y_test, y_pred)
    output_file.write("\nZero-one Loss: " + str(z1l))

    classifier_model = output_path + alg + "_model.sav"
    pickle.dump(model, open(classifier_model, 'wb'))

    if clf_type == 'optimized':
        output_file.write("\nOptimizing Tuple: " + str(optimizing_tuple))
        output_file.write("\nOptimization Time: " + str(optimization_time))
        if class_type == 'binary':
            output_df = pd.DataFrame({'Algorithm': classifier_name_expand(alg),
                                      'Training Time': str(cv_time),
                                      'Fitting Time': str(fitting_time),
                                      'Prediction Time': str(prediction_time),
                                      'Accuracy Score(Test Set)': accScore,
                                      'Accuracy Scores(Training)': " ".join(map(str, acc_scores)),
                                      'Mean Accuracy Score(Training)': acc_scores.mean(),
                                      'Accuracy Score StDev(Training)': acc_scores.std(),
                                      'Precision Score': preScore,
                                      'Recall Score': recScore,
                                      'F1 Score': f1Score,
                                      'F2 Score': f2Score,
                                      'ROC AUC Score': ras, 
                                      'Matthews CC': mcc,
                                      'Cohens Kappa': ck,
                                      'Jaccard Score': jaccard,
                                      'Hamming Loss': hammingl,
                                      'Zero-one Loss': z1l,
                                      'Optimizing Tuple': " ".join(map(str, optimizing_tuple)), 
                                      'Optimization Time': str(optimization_time)
            } ,index=[0])
        else:
            output_df = pd.DataFrame({'Algorithm': classifier_name_expand(alg),
                                      'Training Time': str(cv_time),
                                      'Fitting Time': str(fitting_time),
                                      'Prediction Time': str(prediction_time),
                                      'Accuracy Score(Test Set)': accScore,
                                      'Accuracy Scores(Training)': " ".join(map(str, acc_scores)),
                                      'Mean Accuracy Score(Training)': acc_scores.mean(),
                                      'Accuracy Score StDev(Training)': acc_scores.std(),
                                      'Precision Score': preScore,
                                      'Recall Score': recScore,
                                      'F1 Score': f1Score,
                                      'F2 Score': f2Score,
                                      'Matthews CC': mcc,
                                      'Cohens Kappa': ck,
                                      'Jaccard Score': jaccard,
                                      'Hamming Loss': hammingl,
                                      'Zero-one Loss': z1l,
                                      'Optimizing Tuple': " ".join(map(str, optimizing_tuple)), 
                                      'Optimization Time': str(optimization_time)
            } ,index=[0])   

    else:
        if class_type == 'binary':
            output_df = pd.DataFrame({'Algorithm': classifier_name_expand(alg),
                                    'Training Time': str(cv_time),
                                    'Fitting Time': str(fitting_time),
                                    'Prediction Time': str(prediction_time),
                                    'Accuracy Score(Test Set)': accScore,
                                    'Accuracy Scores(Training)': " ".join(map(str, acc_scores)),
                                    'Mean Accuracy Score(Training)': acc_scores.mean(),
                                    'Accuracy Score StDev(Training)': acc_scores.std(),
                                    'Precision Score': preScore,
                                    'Recall Score': recScore,
                                    'F1 Score': f1Score,
                                    'F2 Score': f2Score,
                                    'ROC AUC Score': ras, 
                                    'Matthews CC': mcc,
                                    'Cohens Kappa': ck,
                                    'Jaccard Score': jaccard,
                                    'Hamming Loss': hammingl,
                                    'Zero-one Loss': z1l
            } ,index=[0])
        else:
            output_df = pd.DataFrame({'Algorithm': classifier_name_expand(alg),
                                    'Training Time': str(cv_time),
                                    'Fitting Time': str(fitting_time),
                                    'Prediction Time': str(prediction_time),
                                    'Accuracy Score(Test Set)': accScore,
                                    'Accuracy Scores(Training)': " ".join(map(str, acc_scores)),
                                    'Mean Accuracy Score(Training)': acc_scores.mean(),
                                    'Accuracy Score StDev(Training)': acc_scores.std(),
                                    'Precision Score': preScore,
                                    'Recall Score': recScore,
                                    'F1 Score': f1Score,
                                    'F2 Score': f2Score,
                                    'Matthews CC': mcc,
                                    'Cohens Kappa': ck,
                                    'Jaccard Score': jaccard,
                                    'Hamming Loss': hammingl,
                                    'Zero-one Loss': z1l
            } ,index=[0])
            
    output_df.to_csv(summary_file_path, index = False)

    return output_path, model

def group_roc_plotter_by_alg(superv_alg_name, output_path, clf_type):
    plt.clf()
    plt.title("ROC Curves by Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
        
    if clf_type == 'optimized':
        for alg in superv_alg_name:
            file_inp = output_path + "/supervised_methods_evaluation_results/" + str(alg) + \
                                        '/optimized_results/' + str(alg) + '_roc_data.csv'  
            fpr = pd.read_csv(file_inp, usecols = ['fpr'])
            tpr = pd.read_csv(file_inp, usecols = ['tpr'])
            thrsh = pd.read_csv(file_inp, usecols = ['thrsh'])
            plt.plot(fpr,tpr,label = classifier_name_expand(alg))
        plt.legend()
        plt.savefig(output_path + "/supervised_methods_evaluation_results/roc_data_optimized_summary.png")
            
    elif clf_type == 'plain':
        for alg in superv_alg_name:
            file_inp = output_path + "/supervised_methods_evaluation_results/" + str(alg) + \
                                        '/plain_results/' + str(alg) + '_roc_data.csv'  
            fpr = pd.read_csv(file_inp, usecols = ['fpr'])
            tpr = pd.read_csv(file_inp, usecols = ['tpr'])
            thrsh = pd.read_csv(file_inp, usecols = ['thrsh'])
            plt.plot(fpr,tpr,label = classifier_name_expand(alg))
        plt.legend()
        plt.savefig(output_path + "/supervised_methods_evaluation_results/roc_data_plain_summary.png")

    plt.close()

def group_summary_by_alg(superv_alg_name, output_path, clf_type, classif_type, output_columns):
    compiled_df = pd.DataFrame(columns=output_columns)

    if classif_type == 'binary':
        if clf_type == 'optimized':
            file_out_path = output_path + "/supervised_methods_evaluation_results/optimized_summary.csv" 
            for alg in superv_alg_name:
                file_inp = output_path + "/supervised_methods_evaluation_results/" + str(alg) + \
                                            '/optimized_results/' + str(alg) + '_summary_optimized.csv'  
                data = pd.read_csv(file_inp, usecols=output_columns, index_col=False)               
                compiled_df = compiled_df.append(data)
                
        elif clf_type == 'plain':
            file_out_path = output_path + "/supervised_methods_evaluation_results/plain_summary.csv" 
            for alg in superv_alg_name:
                file_inp = output_path + "/supervised_methods_evaluation_results/" + str(alg) + \
                                            '/plain_results/' + str(alg) + '_summary_plain.csv'  
                data = pd.read_csv(file_inp, index_col=False)               
                compiled_df = compiled_df.append(data)

    elif classif_type == 'multiclass':
        if clf_type == 'optimized':
            file_out_path = output_path + "/supervised_methods_evaluation_results/optimized_summary.csv" 
            for alg in superv_alg_name:
                file_inp = output_path + "/supervised_methods_evaluation_results/" + str(alg) + \
                                            '/optimized_results/' + str(alg) + '_summary_optimized.csv'  
                data = pd.read_csv(file_inp, index_col=False)
                compiled_df = compiled_df.append(data)
                
        elif clf_type == 'plain':
            file_out_path = output_path + "/supervised_methods_evaluation_results/plain_summary.csv" 
            for alg in superv_alg_name:
                file_inp = output_path + "/supervised_methods_evaluation_results/" + str(alg) + \
                                            '/plain_results/' + str(alg) + '_summary_plain.csv'  
                data = pd.read_csv(file_inp, index_col=False)
                compiled_df = compiled_df.append(data)

    compiled_df.to_csv(file_out_path, index=False)

class dataset():
    def __init__(self):
        self.complete_data = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.train_labels = pd.DataFrame(columns=['label'])
        self.test_data = pd.DataFrame()
        self.test_labels = pd.DataFrame(columns=['label'])
        self.train_data_2d = []
        self.test_data_2d = []
        self.classif_type = ''
        self.data_path = ''
        self.data_sources = []
        self.scaled = ''

    def scale_data(self, scaling):
        if scaling == 'standard':
            self.train_data[[col for col in self.train_data]] = StandardScaler().fit_transform(self.train_data[[col for col in self.train_data]], self.test_data)
        elif scaling == 'min_max':
            min_max_scaler = MinMaxScaler()
            self.train_data[[col for col in self.train_data]] = min_max_scaler.fit_transform(self.train_data[[col for col in self.train_data]], self.test_data)
        else:
            print("Entered scaling arg not supported")

    def select_data(self):
        root = tk.Tk()
        root.withdraw()
        while len(self.data_sources) == 0:
            self.data_sources = filedialog.askopenfilename(title="Select data file(s)", filetypes=[("csv", ".csv")], multiple = True)
            print("Select at least one source of data" if (len(self.data_sources) == 0) else self.data_sources)
        self.data_path = dirname(self.data_sources[0])
        root.quit()

    def load_data(self):
        temp_df = pd.DataFrame()
        for file in self.data_sources:
            data = pd.read_csv(file)
            temp_df = pd.concat([temp_df, data], ignore_index=True)

        temp_df = pd.DataFrame(sk.utils.shuffle(temp_df))      
        self.complete_data = temp_df

        if 'label' in data:
            self.train_data = pd.concat([self.train_data, temp_df.loc[:, temp_df.columns != 'label']], ignore_index=True)
            self.train_labels = pd.concat([self.train_labels, pd.DataFrame(temp_df['label'])], ignore_index=True) 
        else:
            self.train_data = pd.concat([self.train_data, temp_df], ignore_index=True)
            self.classif_type = 'unlabeled'    
        self.get_classif_type()

    ### add for semi-supervised/unsupervised case, add proper error handling 
    def get_classif_type(self):
        if self.classif_type != 'unlabeled':
            label_count = len(self.train_labels.label.unique())
            if label_count == 2:
                self.classif_type = 'binary'
            elif label_count > 2:
                self.classif_type = 'multiclass'
            else:
                print("Number of labels not currently supported")

    def split_data(self, split_size):
        if self.classif_type != 'unlabeled':
            self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.train_data,
                                                                                                    self.train_labels,
                                                                                                    test_size=split_size,
                                                                                                    random_state = 42)
        else:
            self.train_data, self.test_data = train_test_split(self.train_data, test_size=split_size, random_state=42)

    def downsize_data(self):
        print("Downsizing data")
        self.train_data_2d = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(self.train_data)
        self.test_data_2d = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(self.test_data)
        print("Finished downsizing data")

class learning():
    def __init__(self, data_obj, arg_type, selected_algs=[]):
        self.data_obj = data_obj
        self.arg_type = arg_type
        self.selected_algs = selected_algs
        self.summary_cols = self.get_summary_column_labels(self.data_obj.classif_type, self.arg_type)

    def get_bin_classif_algs(self):

                #lsvc and svc take long
                #gnb, mnb, and compnb cannot take negative values
                #giving errors see notes: 'dt', 'gp' 
        self.selected_algs = ['rf', 'hgbc',  'ada', 'qda', 'lda', 'mlp', 'ridge',
                              'pa', 'sgd', 'etc'] #, 'gnb', 'mnb', 'knn', 'svc', 'gp', 'lsvc', 'dt']
        
    def get_multi_classif_algs(self):
        self.selected_algs = ['gnb', 'bnb', 'lda', 'lsvc', 'qda', 'ridge',
                              'rf', 'hgbc', 'mlp' ] #, 'knn']
        
    def get_summary_column_labels(self, classif_type, arg_type):
        summary_cols_coll = { ('binary', 'optimized'): [ 'Algorithm', 'Training_Time', 'Fitting_Time', 
                       'Prediction_Time', 'Accuracy_Score(Test_Set)',
                       'Accuracy_Scores(Training)', 'Mean_Accuracy_Score(Training)',
                       'Accuracy_Score_StDev(Training)', 'Precision_Score',
                       'Recall_Score', 'F1_Score', 'F2_Score',
                       'ROC_AUC_Score', 'Matthews_CC', 'Cohens_Kappa',
                       'Jaccard_Score', 'Hamming_Loss', 'Zero-one_Loss', 
                       'Optimizing_Tuple', 'Optimization_Time'], 
                       
                       ('binary', 'plain'): [ 'Algorithm', 'Training_Time', 'Fitting_Time', 
                       'Prediction_Time', 'Accuracy_Score(Test_Set)',
                       'Accuracy_Scores(Training)', 'Mean_Accuracy_Score(Training)',
                       'Accuracy_Score_StDev(Training)', 'Precision_Score',
                       'Recall_Score', 'F1_Score', 'F2_Score',
                       'ROC_AUC_Score', 'Matthews_CC', 'Cohens_Kappa',
                       'Jaccard_Score', 'Hamming_Loss', 'Zero-one_Loss'],
                       
                       ('multiclass', 'optimized'): [ 'Algorithm', 'Training_Time', 'Fitting_Time', 
                       'Prediction_Time', 'Accuracy_Score(Test_Set)',
                       'Accuracy_Scores(Training)', 'Mean_Accuracy_Score(Training)',
                       'Accuracy_Score_StDev(Training)', 'Precision_Score',
                       'Recall_Score', 'F1_Score', 'F2_Score',
                       'Matthews_CC', 'Cohens_Kappa',
                       'Jaccard_Score', 'Hamming_Loss', 'Zero-one_Loss'
                       'Optimizing_Tuple', 'Optimization_Time'],

                       ('multiclass', 'plain'): [ 'Algorithm', 'Training_Time', 'Fitting_Time', 
                       'Prediction_Time', 'Accuracy_Score(Test_Set)',
                       'Accuracy_Scores(Training)', 'Mean_Accuracy_Score(Training)',
                       'Accuracy_Score_StDev(Training)', 'Precision_Score',
                       'Recall_Score', 'F1_Score', 'F2_Score',
                       'Matthews_CC', 'Cohens_Kappa',
                       'Jaccard_Score', 'Hamming_Loss', 'Zero-one_Loss'] 
                       }
        
        self.summary_cols = summary_cols_coll[(classif_type, arg_type)]
        
    def supervised_learning(self):
        if self.data_obj.classif_type == 'binary' and self.selected_algs == []:
            self.get_bin_classif_algs()
        elif self.data_obj.classif_type == 'multiclass' and self.selected_algs == []:
            self.get_multi_classif_algs()
        
        for alg in self.selected_algs:
            self.classif_funcs_caller(alg)

        group_summary_by_alg(self.selected_algs, self.data_obj.data_path, self.arg_type, self.data_obj.classif_type, self.summary_cols)

        if self.data_obj.classif_type == 'binary':

            group_roc_plotter_by_alg(self.selected_algs, self.data_obj.data_path, self.arg_type)
    
        #print("Learning step completed")

    def classif_funcs_caller(self, alg): 
         
        X_train_size = len(self.data_obj.train_data)
        y_train_size = len(self.data_obj.train_labels)
        data_labels = self.data_obj.train_labels.label.unique()

        if self.arg_type == 'optimized':
            print("Current Alg: ", alg)
            model = plain_clf_runner(alg)
            model, optimizing_record, optimizing_time = supervised_learning_caller_optimized(alg, self.data_obj.train_data,
                                                                                                  self.data_obj.train_labels, 
                                                                                                  self.data_obj.test_data,
                                                                                                  self.data_obj.test_labels)

            output_path1, model = supervised_methods_evaluation(alg, model,
                                                                self.data_obj.train_data,
                                                                self.data_obj.train_labels, 
                                                                self.data_obj.test_data,
                                                                self.data_obj.test_labels,
                                                                self.data_obj.train_data_2d,
                                                                self.data_obj.test_data_2d,
                                                                X_train_size, y_train_size,
                                                                self.data_obj.data_path, 
                                                                data_labels, self.arg_type,
                                                                self.data_obj.classif_type,
                                                                optimizing_record, optimizing_time
                                                                )

        elif self.arg_type == 'plain':
            print("Current Alg: ", alg)
            model = plain_clf_runner(alg)
            output_path1, model = supervised_methods_evaluation(alg, model,
                                                                self.data_obj.train_data,
                                                                self.data_obj.train_labels, 
                                                                self.data_obj.test_data,
                                                                self.data_obj.test_labels,
                                                                self.data_obj.train_data_2d,
                                                                self.data_obj.test_data_2d,
                                                                X_train_size, y_train_size,
                                                                self.data_obj.data_path, 
                                                                data_labels, self.arg_type,
                                                                self.data_obj.classif_type)
        return model





