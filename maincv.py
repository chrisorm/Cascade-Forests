#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:35:23 2017

@author: chris
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepForestCV import deepForestCv

X,y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60)

df = deepForestCv()
df.fit(X_train, y_train)

base = RandomForestClassifier(n_estimators=2000,min_samples_leaf=10)
base.fit(X_train, y_train)


mypred= df.predict(X_test).ravel()
print "Deep Forest:",accuracy_score(y_test, mypred)
print "RF: ",accuracy_score(y_test, base.predict(X_test))