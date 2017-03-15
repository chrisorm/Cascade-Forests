#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:03:17 2017

@author: chris
"""

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier

class deepForestLayer(ClassifierMixin):
    def __init__(self, n_nodes):
        nrfs = int(n_nodes/2) #number of random forests
        nefs = n_nodes - nrfs #number of extra random trees
        self.estimators = []
        for i in range(nefs):
            self.estimators.append(('ET'+str(i),ExtraTreesClassifier(n_estimators=1000,min_samples_leaf=10, n_jobs=-1)))
            
        for i in range(nrfs):
            self.estimators.append(('RF'+str(i),RandomForestClassifier(n_estimators=1000,min_samples_leaf=10,n_jobs=-1)))
            
        self.voter = VotingClassifier(estimators=self.estimators, voting='soft') #use a voting classifier to combine.
        
    def fit(self, X,y):
        """Simple wrapper function around votingclassifier"""
        self.voter.fit(X,y)
    
    def predict(self, X):
        """Simple wrapper function around votingclassifier"""
        return self.voter.predict(X)
        
    def predict_proba(self, X):
        """Simple wrapper function around votingclassifier"""
        return self.voter.predict_proba(X)
    
    def transform(self,X):
        """Simple wrapper function around votingclassifier. This is useful for the multigrain scanning section"""
        return self.voter.transform(X)
    
