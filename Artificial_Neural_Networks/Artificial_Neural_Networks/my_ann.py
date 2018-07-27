#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 09:28:11 2018

@author: ekele
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense

#classifier = Sequential()
#
#classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_shape = (11,)))
#
#classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
#
#classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
#classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#
#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
#
##sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]]))
#
#new_predictions = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
#new_predictions = (new_predictions > 0.5)
#
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

'''
K Fold cross validation
KerasClassifier wrapper
'''

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_shape = (11,)))
    classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32], 
              'epochs': [100, 500],
              'optimizer': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

best_accuracy







