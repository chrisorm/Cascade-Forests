#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:23:01 2017

@author: chris
"""

from sklearn.model_selection import cross_val_score
from deepforest import deepForest
from copy import deepcopy
import numpy as np

class deepForestCv(object):
    
    def __init__(self):
        pass
    
    def fit(self, X,y):
        """implement the layer size cross validation method"""
        
        last_score = -np.inf
        for i in range(1, max_layers+1):
            new_clf = deepForest(n_layers=i)
            new_score=np.mean(cross_val_score(new_clf, X, y))
            print "Mean Score:", new_score, " with "+ str(i) + " layers"
            if new_score <=last_score:
                break
            last_score=new_score
        i=i-1
        print "Fitting with "+str(i)+" layers..."
        self.estimator = deepForest(n_layers=i)
        self.estimator.fit(X,y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
