#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:29:36 2017

@author: pratheekdevaraj
"""

#%%
#import data and libs
import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#load data
iris = pd.read_csv('data/iris.csv')

#%%

#view data


#shuffle data
iris.apply(np.random.shuffle)

iris.head(10)


#%%

X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
le = LabelEncoder()
le.fit(iris[['species']])
#print(le.classes_)
y = le.transform(iris[['species']])

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#%%

#logistic model

logistic = LogisticRegression(C=1e5)

logistic.fit(X_train,y_train)

#predict
print('predicted values:')
logistic.predict(X_test)
print('score:')
logistic.score(X_test, y_test)
