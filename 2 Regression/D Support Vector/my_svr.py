# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 07:52:40 2018

@author: USER
"""

# lecture 67 - SVR from template
# SVR disregards errors within epsilon value

# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
# need to reshape Y array for feature scaling so changed col from 2 to 2:3
# possible alt is y = y.reshape(-1,1)


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling -- NEEDED FOR SVR (aabout 6:30, L67)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
# gives "change int 64 to float 64" warning.  no biggie

# !!!
# Fitting thSVR to dataset
# !!!

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
# kernel: linear, polynomial, gaussian, rbf (default but good to specify)
# also set epsilon val, etc.
regressor.fit(X, y)
# gives output

# !!! that's it!

# Predicting a new result / edits made around 15 -1 7 mark
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) 
# need to change 6.5 to transformed value b/c of feature scaling
# but transform expects an array so 6.5 cannot be data type
#use numpy array to create 1 cell array, two pairs of brackets makes array 1x1 (one pair = vector)
# need to use inverse transform on y to then get orig scale of salary



# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# or can use "smoother" code for smoother curve
