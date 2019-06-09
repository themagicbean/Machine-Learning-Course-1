# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 23:40:45 2018

@author: USER
"""
# L 69 DECISION TREE 
# L70 

#copypasta from template

# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set - n/a b/c small set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the DT Regression Model to the dataset !!! ---- !!!
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
#gives output

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Regression results -- using low res = problem
# red flag -- model seems to perfect.  so no actual predictions ...
# is predicting each of 10 levels, then joining by straight line 
# but this is noncontinuous model - change to higher res 

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (DTR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

