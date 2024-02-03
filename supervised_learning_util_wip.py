import numpy as np
import pandas as pd

import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from os.path import dirname

from classifier_helper import *
from supervised_learning_eval import *
from multiprocessing import cpu_count

import tkinter as tk
from tkinter import filedialog

from concurrent.futures import ThreadPoolExecutor

#import warnings
#warnings.filterwarnings("ignore")

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
            data = pd.read_csv(file, index_col=0)
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
        summary_cols_coll = { ('binary', 'optimized'): [ 'Algorithm', 'Training Time', 'Fitting Time', 
                       'Prediction Time', 'Accuracy Score(Test Set)',
                       'Accuracy Scores(Training)', 'Mean Accuracy Score(Training)',
                       'Accuracy Score StDev(Training)', 'Precision Score',
                       'Recall Score', 'F1 Score', 'F2 Score',
                       'ROC AUC Score', 'Matthews CC', 'Cohens Kappa',
                       'Jaccard Score', 'Hamming Loss', 'Zero-one Loss', 
                       'Optimizing Tuple', 'Optimization Time'], 
                       
                       ('binary', 'plain'): [ 'Algorithm', 'Training Time', 'Fitting Time', 
                       'Prediction Time', 'Accuracy Score(Test Set)',
                       'Accuracy Scores(Training)', 'Mean Accuracy Score(Training)',
                       'Accuracy Score StDev(Training)', 'Precision Score',
                       'Recall Score', 'F1 Score', 'F2 Score',
                       'ROC AUC Score', 'Matthews CC', 'Cohens Kappa',
                       'Jaccard Score', 'Hamming Loss', 'Zero-one Loss'],
                       
                       ('multiclass', 'optimized'): [ 'Algorithm', 'Training Time', 'Fitting Time', 
                       'Prediction Time', 'Accuracy Score(Test Set)',
                       'Accuracy Scores(Training)', 'Mean Accuracy Score(Training)',
                       'Accuracy Score StDev(Training)', 'Precision Score',
                       'Recall Score', 'F1 Score', 'F2 Score',
                       'Matthews CC', 'Cohens Kappa',
                       'Jaccard Score', 'Hamming Loss', 'Zero-one Loss', 
                       'Optimizing Tuple', 'Optimization Time'],

                       ('multiclass', 'plain'): [ 'Algorithm', 'Training Time', 'Fitting Time', 
                       'Prediction Time', 'Accuracy Score(Test Set)',
                       'Accuracy Scores(Training)', 'Mean Accuracy Score(Training)',
                       'Accuracy Score StDev(Training)', 'Precision Score',
                       'Recall Score', 'F1 Score', 'F2 Score',
                       'Matthews CC', 'Cohens Kappa',
                       'Jaccard Score', 'Hamming Loss', 'Zero-one Loss'] 
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
    
        print("Learning step completed")
    
        return 0

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

def main():
    
    ### run on binary data
    data = dataset()
    data.select_data()
    data.load_data()
    data.split_data(.2)
    data.downsize_data()
    learning_obj = learning(data, 'plain', ['hgbc',  'ada', 'qda', 'lda'])
    learning_obj.supervised_learning()
    
    learning_obj = learning(data, 'optimized', ['hgbc',  'ada', 'qda', 'lda'])
    learning_obj.supervised_learning()

    ### run on multiclass data  

    data = dataset()
    data.select_data()
    data.load_data()
    data.split_data(.2)
    data.downsize_data()
    learning_obj = learning(data, 'plain', ['hgbc',  'qda', 'ridge', 'lda'])
    learning_obj.supervised_learning()
    
    learning_obj = learning(data, 'optimized', ['hgbc',  'qda', 'ridge', 'lda'])
    learning_obj.supervised_learning()
    
    return 0

main()





