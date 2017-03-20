#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 07:53:46 2017

@author: chris
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepforest import DeepForest
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('data/letter-recognition.data', header=None)
y = df[0].get_values()
X = df.drop([0], axis=1).get_values()

le = preprocessing.LabelEncoder()
y=le.fit_transform(y)

print "data read in ok...."
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

df = DeepForest()
df.fit(X_train, y_train)

base = RandomForestClassifier()
gs = GridSearchCV(base, {'n_estimators':[2000]})
gs.fit(X_train, y_train)


mypred= df.predict(X_test).ravel()
print "Deep Forest:",accuracy_score(y_test, mypred)
print "RF: ",accuracy_score(y_test, gs.predict(X_test))

with open('letters.txt', 'w') as f:
    f.write("DeepForest:"+str(accuracy_score(y_test, mypred))+"\n bench:"+str(accuracy_score(y_test, gs.predict(X_test))) )
