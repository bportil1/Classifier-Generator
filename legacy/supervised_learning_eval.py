import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.model_selection import GridSearchCV
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

import warnings
warnings.filterwarnings("ignore")

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
    acc_scores = cross_val_score(model, X, y, cv = 5, n_jobs = -1)
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
        
    return 0

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
 
    return 0

    
