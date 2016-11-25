#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:06:44 2016

@author: dgt377
"""
#%%
C_range = np.array([.01, 1, 5, 10, 20, 50, 100, 1000, 5000, 10000])
gamma_range = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])

storeCV = np.zeros((len(gamma_range),len(C_range)))
storeTrain = np.zeros((len(gamma_range),len(C_range)))

for outer_ind, gamma_value in enumerate(gamma_range):
        cv_errors = np.zeros(C_range.shape)
        train_errors = np.zeros(C_range.shape)
        for index, c_value in enumerate(C_range):
            clf = svm.SVC(C=c_value, gamma=gamma_value)
            clf.fit(X_train,y_train)
            
            train_conf = confusion_matrix(y_train, clf.predict(X_train))
            cv_conf = confusion_matrix(y_test, clf.predict(X_test))
        
            cv_errors[index] = accuracy(cv_conf)
            train_errors[index] = accuracy(train_conf)
        storeCV[outer_ind,:] = cv_errors
        storeTrain[outer_ind,:] = train_errors

b=np.where(storeCV==storeCV.max())
best_C=C_range[b[0][0]]
best_gamma=gamma_range[b[1][0]]

#%%
A=np.arange(10)
A[3]=NaN