#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:24:45 2017

@author: chris
"""

from rflayer import deepForestLayer
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator

class deepForest(ClassifierMixin,BaseEstimator):
    def __init__(self, n_layers=1, width=4):
        self.n_layers=n_layers
        self.width= width
        self.layers=[]
        self.layerScores=[]
    
    def fit(self, X,y):
        for layer in range(self.n_layers):
            self.layers.append(deepForestLayer(self.width))
        previous_pred = np.empty((X.shape[0],0))
        for layer in self.layers:
            newX=np.concatenate([X,previous_pred], axis=1)
            layer.fit(newX,y)
            previous_pred = layer.predict_proba(newX)
            
            
    def predict_proba(self, X):
        previous_pred = np.empty((X.shape[0],0))
        for layer in self.layers:
            newX=np.concatenate([X,previous_pred], axis=1)
            previous_pred = layer.predict_proba(newX)
        return previous_pred
    
    def predict(self,X):
        return np.argmax(self.predict_proba(X), axis=1)