# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 02:40:27 2018

@author: USER
"""

#lecture 85 logistic regression in py

#copypasta data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset - will need to change names and columnts
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values #only analyzing age and salary
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split #cross_validation deprecated
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling - commentized bc generally includid in ML libs
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# geneartes warning that int64 data converted to float

#L 86 fitting model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
#gives output

#L 87 predicting test set results
y_pred = classifier.predict(X_test)

#L 88 makinig the confusion matrix (evaluates predictive power)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # may also want to use labels ? ...
#type cm in console to view it (0,01 and 1,1 are correct 0,1 and 1,0 are not)

# L 89
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# two color regions are prediction regions, separated by prediction boundary
# straignt boundary because linear classifier. but users not linearly distributed. so not perfect.  woo
# colors, though ugly, better than most other combos I tried.  no hard edges on dots
# these boundaries will not change unless classifier is altered.  

# and the test set results - change train to test
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test # shortcut to avoid having to retype test/traini over and over
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# add -1 and +1 to min and max to avoid squeezing points on pixel display
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#contour does the colour regions
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
#scatterplot does dots
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# a few incorret predictions but whoop dee doo




