#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:16:58 2017

@author: chris
"""

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.model_selection import KFold
import numpy as np
import copy
class deepForestLayer(ClassifierMixin):
    """A Really hacky and WIP implementation of the layers required for a Cascade Forest"""
    def __init__(self, n_nodes, nClasses=1,output=False):
        self.output=output
        self.nClasses = nClasses
        self.n_nodes= n_nodes
        nrfs = int(n_nodes/2)
        nefs = n_nodes - nrfs
        self.estimators = []
        self.final_voters = []
        for i in range(nefs):
            self.estimators.append(('ET'+str(i),ExtraTreesClassifier(n_estimators=1000,min_samples_leaf=10, n_jobs=-1)))
            
        for i in range(nrfs):
            self.estimators.append(('RF'+str(i),RandomForestClassifier(n_estimators=1000,min_samples_leaf=10,n_jobs=-1)))
        self.voter = VotingClassifier(estimators=self.estimators, voting='soft')

    def fit_Kfold(self,X_train, X_test,y_train, y_test):
        """This function implements the growing and validation to determine number of layers required"""
        
        fold = KFold() # 3-Fold CV
        train_preds = np.empty((3,X_train.shape[0],self.nClasses*self.n_nodes))
        train_preds[:] = np.nan
        est_preds = np.empty((3,X_test.shape[0],self.nClasses*self.n_nodes))
        est_preds[:] = np.nan
        i=0
        for train_idx, test_idx in fold.split(X_train):
            self.voter.fit(X_train[train_idx],y_train[train_idx]) #Fit each of the estimators to our data
            
            #voter.transform has shape number of estimators (4 in this case) x no of samples x number of classes.
            #Insample is the transform output reshaped to have shape no of samples x number of classes*number of estimators
            insample = self.voter.transform(X_train[train_idx]).swapaxes(0,1).reshape((X_train[train_idx].shape[0],-1))
            outsample = self.voter.transform(X_test).swapaxes(0,1).reshape((X_test.shape[0],-1))
            #Insample is the training error, outsample is the validation error.
            
            train_preds[i,train_idx] = insample.copy()
            est_preds[i] = outsample.copy()
            
            i+=1
        #As I used KFold, train_preds and est_preds have two valid entries and one nan entry per data point
        #average this dimension so we get one probability prediction per data point
        return np.nanmean(train_preds,axis=0),np.nanmean(est_preds,axis=0)
        
    def fit(self, X,y):
        """This function does a full fit once the number of layers has been decided"""
        fold = KFold()
        #Create 3 models, each fitted on a fold of the data.
        #This is only way I can think of getting required output at prediction stage.
        for train_idx, test_idx in fold.split(X):
            clf=VotingClassifier(estimators=copy.deepcopy(self.estimators), voting='soft')
            clf.fit(X[train_idx], y[train_idx])
            self.final_voters.append(clf)
        
    def predict(self, X):
        """Make predictions, using models fitted by KFold"""
        preds = np.zeros((len(self.final_voters),X.shape[0],self.nClasses))
        for i in range(len(self.final_voters)):
            preds[i]=self.final_voters[i].predict_proba(X)
        return np.mean(preds,axis=0)
        
    