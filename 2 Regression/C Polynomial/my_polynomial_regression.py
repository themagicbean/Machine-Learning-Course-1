# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 06:04:37 2018

@author: USER
"""

# lecture 55 polynom reg in py

# data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# dataset column 1 (level) is basically encoding of position (col 0)

X = dataset.iloc[:, 1:2].values # if only one index produces a vector (undesirable)
# previously used X = dataset.iloc[:, 1].values, which produced a vector
# 1:2 produces matrix, top bound is excluded, so no diff in data

********* 8: 44 in 55 *********

y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set -- NOT NEEDED, sample too small
"""from sklearn.cross_validation import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling -- ALSO NOT NEEDED, library does it auto
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# L56 - fit linear and polynomial models to data set
# linear first
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# gives output

# fit poly regress model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
# this was 2 initially, changed to 3 in L57 about 15' then 4 at about 17'
X_poly = poly_reg.fit_transform(X)
# transforms X into X with x^1, X^2, ... (to higher powers as desired - degree arg)
# X_poly is name for transformed matrix\
# bias arg in PF auto creates a constant column (all 1s) to act as intercept/constant

#create 2d lin reg object to include fit and poly 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# gives output

#L57
# visualize LR predictions = make a graphola.  
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff? Linear Regression')
plt.xlabel('Position Level')
ply.ylabel('Salary')
plt.show()

# visualize PR precitions
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
# can't just replace lin_reg w/ lin_reg_2 because it is lin reg class
#don't just change X to X_poly. b/c X_poly already defined.  
# as typed, easier to change to new MOF (change all Xs to whatever)
plt.title('Truth or Bluff? Polynomial Regression')
plt.xlabel('Position Level')
ply.ylabel('Salary')
plt.show()

# add degree to PR model - change degree in PR code (line 51)
# added 2d degree (total 4) to make most accurate model

# can still improve -- seek continuous curve by using increments of higher res
# i.e., 0.1 step 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
# reshapes X_grid into matrix
# then change X to X_grid in plot (not in scatter b/c scatter is original data)

#L58 - comparing value (employee asserted past salary at level 6.5) to prediction (by model)
lin_reg.predict(6.5)
# gives output of about 330k - way above assertion of 160 by employee
lin_reg_2.predict(poly_reg.fit_transform(6.5))
# gives 158 k, close to assertion by employee

# lecture 59 gives python template, already in folder as regression_template.py

























