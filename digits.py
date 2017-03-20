#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:37:57 2017

@author: chris
"""

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepforest import DeepForest

X,y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

df = DeepForest()
df.fit(X_train, y_train)

base = RandomForestClassifier()
gs = GridSearchCV(base, {'n_estimators':[2000]})
gs.fit(X_train, y_train)


mypred= df.predict(X_test)
print "Deep Forest:",accuracy_score(y_test, mypred)
print "RF: ",accuracy_score(y_test, gs.predict(X_test))

