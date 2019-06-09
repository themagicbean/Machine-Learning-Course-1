# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 01:57:33 2018

@author: USER
"""

# data preprocessing
# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset; specify wd first (use file explorer)
# set wd to folder taht contains .csv data file
# on win, save py file into folder that contains.csv file
# then run file (green arrow) or f5 as shortcut to run * set wd

#use pandas to import data set

dataset = pd.read_csv('Data.csv')

#doubleclick on dataset in variable explorer to see set

# can change float/etc in data set
# change/create matrix of features (matrix of ind variables)
X = dataset.iloc[:, :-1].values
#: = all -1 saves the last (which will be dep variable) .values gets values

# create dep variable vector
y = dataset.iloc[:, 3].values
#they use lowercase for y and upper for X???

#lecture 12 / taking care of missing data
from sklearn.preprocessing import Imputer

# create object of the class of imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
# ctrl I to inspect Imputer gives you parameters in help window
#fit imputer object to matrix X

imputer = imputer.fit(X[:, 1:3])
#upper bound excludes, so to take 1 and 2 you type 1:3
# imputer object fitted to matrix X
# then fill in with mean

X[:, 1:3] = imputer.transform(X[:, 1:3])
#after this if display X in console, missing data replaced by means

#lecture 14, categorical variables - change text to values  (encode)
from sklearn.preprocessing import LabelEncoder

#create instance / object
labelencoder_X = LabelEncoder()
# xxx labelencoder_X.fit_transform(X[:, 0])
#fitted labelencoder to 1st column (country) and returns it encoded
# then changed line XXX to 
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# this actually encodes the dataset

#!!!
# !!!problem: 2 > 1 > 0 arithmatecially.  need to de-relation order

#dummy encoding - make # of columns = to number of categories
# then each is 0/1 
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# did not need to specify column b/c specd [0] in line above
# to code purchase value, d/n need onehotencoder, no relative val

# !!!
# cross_validation given in lecture replaced with model_selection
# !!!

# lecture 16 splitting the dataset 
# need to split into training and test sets 
# performance on test set should be similar to training set
# shows adaptability in models

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

# test size is % of data to train model (usually around 20%)
# test_size + train_size= 1 so only do one
# random_state set to same # gives same random on diff machines (?)
# creates X & y train & test in variable explorer 
# "overfitting" when can n/ perform well b/c too "close" to training materials

# 17 feature scaling
#e.g., note age and salary variable have different scales
# b/c many models based on euclidian distance
# standardization  X - meanX / std dev X
# normalisation x - min X / max X - min X

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# training fit & transform.  test just transform?  
#bc already fitted to training set
X_test = sc_X.transform(X_test)
# do you need to scale dummy variables?  maybe.  whee.
# or to dep var vector (y)? no b/c dep var categorical here
# if dep var could hvae range, then feature scaling would be needed

# most libraries d/n require manual feature scaling 










