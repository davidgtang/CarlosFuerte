#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:03:00 2016

@author: kdarnell
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm, display_adj_cm
from pandas import set_option
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.model_selection import LeaveOneGroupOut, validation_curve





set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

filename = 'training_data.csv'
training_data = pd.read_csv(filename)
# %%
blind = training_data[training_data['Well Name'] == 'SHANKLE']
training_data = training_data[training_data['Well Name'] != 'SHANKLE']
# %%
facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D', 'PS', 'BS']


def label_facies(row, labels):
    return labels[row['Facies'] - 1]
    
# "df.apply" with a lambda function returns iloc with an iterator that can be accessed like this: "training_data.iloc[1]['Facies']" which is equal to "training_data['Facies'][1]", but in the former case all columns are available to the lambda function.
training_data.loc[:,'FaciesLabels'] = training_data.apply(lambda row: label_facies(row, facies_labels), axis=1)
# %%
feature_vectors = training_data.drop(['Formation', 'Well Name', 'Depth','Facies','FaciesLabels'], axis=1)#,'FaciesLabels'
# %%
scaler = preprocessing.StandardScaler().fit(feature_vectors)
scaled_features = scaler.transform(feature_vectors)
# %%
pca = PCA(n_components=7,svd_solver='full')
scaled_featurespca = pca.fit_transform(scaled_features)
weight = abs(pca.components_[1,:])
# %%
#X_train, X_test, y_train, y_test = train_test_split(
#        scaled_features, correct_facies_labels, test_size=0.1, random_state=42)
X_train = scaled_features
y_train = correct_facies_labels
# %%

SVC_classifier = svm.SVC(cache_size = 800, random_state=1)

Fscorer = make_scorer(f1_score, average = 'micro')
Ascorer = make_scorer(accuracy_score)
# %%
col_names = feature_vectors.columns

class_dict1 = {}
class_dict2 = {}
for ind,col in enumerate(col_names):
    if col=='PE':
        class_dict1[ind+1] = 2
    else:
        class_dict1[ind+1] = 1
    class_dict2[ind+1] = weight[ind]
# %%
parm_grid={'kernel': ['linear', 'rbf'],
            'C': [0.5, 1, 5, 10, 15],
            'gamma':[0.0001, 0.001, 0.01, 0.1, 1, 10],
            'class_weight':[class_dict1,class_dict2,None]}
#parm_grid={'kernel': ['rbf'],
#            'C': [1, 5],
#            'gamma':[ 0.1, 1],
#            'class_weight':[class_dict1,class_dict2]}

grid_search = GridSearchCV(SVC_classifier,
                           param_grid=parm_grid,
                           scoring = Fscorer,
                           cv=10) # Stratified K-fold with n_splits=10
                                  # For integer inputs, if the estimator is a
                                  # classifier and y is either binary or multiclass,
                                  # as in our case, StratifiedKFold is used
            
grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

grid_search.best_estimator_
# %%
conf = confusion_matrix(y_test, predicted_labels,np.arange(1,10))
display_cm(conf, facies_labels, hide_zeros=True)