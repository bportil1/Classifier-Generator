import numpy as np
import pandas as pd

import scipy.spatial.distance
from scipy import spatial

import sklearn as sk
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import *
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import discriminant_analysis
from sklearn.model_selection import cross_val_score

import time
from datetime import timedelta

from concurrent.futures import ThreadPoolExecutor

from supervised_learning_eval import *
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
                                                              n_jobs=-1).fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    curr_score = sk.metrics.accuracy_score(y_test, y_pred)
                else:
                    if n == 'kd_tree' and metric_passed in [ scipy.spatial.distance.canberra, scipy.spatial.distance.chebyshev, scipy.spatial.distance.correlation, scipy.spatial.distance.sqeuclidean, ]:
                        continue
                    model = sk.neighbors.KNeighborsClassifier(n_neighbors=j,
                                                              weights=m,
                                                              algorithm=n,
                                                              metric=metric_passed,
                                                              n_jobs=-1).fit(X_train, y_train)
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
                model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                                    n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                                        n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                                        n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                                n_jobs=-1) 
        print("Fitting model: ", kernel_passed)
        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                                    n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                                    n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                                    n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                                    n_jobs=-1) 
            print("Fitting model: ", kernel_passed)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                               n_jobs=-1)
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
        model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                               n_jobs = -1,
                                                               random_state = 25)
            model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                          n_jobs = -1,
                                                          random_state = 25,
                                                          learning_rate=j,
                                                          eta0=eta0)
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                               n_jobs=-1)
                    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
                                                  n_jobs=-1)
    model_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = -1 )
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
