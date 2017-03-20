#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:00:17 2017

@author: chris
"""

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from forestlayer import deepForestLayer
from sklearn.metrics import accuracy_score

class DeepForest(ClassifierMixin,BaseEstimator):
    def __init__(self, width=4):
        self.width= width
        self.layers=[]
        self.layerScores=[]

    def fit(self, X,y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        n=len(np.unique(y))
        last_score = -np.inf
        previous_est =  np.empty((X_test.shape[0],0))
        previous_pred = np.empty((X_train.shape[0],0))
        i=1
        while True:
            print "layer "+str(i)
            layer = deepForestLayer(4,nClasses=n)
            pred, est = layer.fit_Kfold(np.concatenate([X_train,previous_pred],axis=1), np.concatenate([X_test,previous_est],axis=1), y_train,y_test)
            yest = np.argmax(np.mean(est.reshape((X_test.shape[0],-1, n)).swapaxes(0,1),axis=0),axis=1)
            score = accuracy_score(y_test, yest)
            print score
            if score<=last_score:
                break
            last_score=score
            previous_est=est
            previous_pred=pred
            i+=1
        print "Selected "+str(i-1)+" layers"
        self.full_fit(X,y,i-1)
        
    def full_fit(self, X,y, layers):
         n=len(np.unique(y))
         previous_pred = np.empty((X.shape[0],0))
         for l in range(layers):
             layer = deepForestLayer(4, nClasses=n)
             newX = np.concatenate([X,previous_pred],axis=1)
             layer.fit(newX,y)
             previous_pred = layer.predict(newX)
             self.layers.append(layer)
         print "fitted!!"
             
    def predict(self, X):
        previous_pred = np.empty((X.shape[0],0))
        for layer in self.layers:
            newX = np.concatenate([X,previous_pred],axis=1)
            previous_pred = layer.predict(newX)
        return np.argmax(previous_pred,1)
            
            
    