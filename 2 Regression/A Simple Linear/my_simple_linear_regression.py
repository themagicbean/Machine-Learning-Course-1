# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 08:12:59 2018

@author: USER
"""

# lecture 24 linear regression
# set wd properly ... i hope

#step 1 preprocess via template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset -- CHANGE NAME
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#no feature scaling needed for simple linear regression

# lecture 25 fitting model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# note gives output stating parameters in LR no biggie

# create object of linear regressor class, to be fit to train set via 
# fit method (of several methods)

#predicting test set results ... (lecture 26)
# vector of predictions of dep (y) var

y_pred = regressor.predict(X_test)
# now you have y pred in variables explorer to compare w y test vals

# lecture 27 visualising the results -- use matplotlib 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# made pred X val from train the X vals to plot - plots regression line
# next few lines label title, axes, etc
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Dat Moolah')
plt.show
#plt.show says "ready to plot the graph" =-> gives a graphical output in console

#now for the test set :: change "train" to "test" derp
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#no change needed in 2d line bc line already trained on training data
#you want the same line/predictions
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Dat Moolah')
plt.show
