# Random Forest Regression

# Lecture 74 
#copypasta
# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Forest Model !!! --- !!! start here
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
# can do more than 10 trees  /less than 300 in the forest (preferred)
# but do not expect # of steps to increase = to # of tree increase
# because steps represent average in a certain range/state
regressor.fit(X, y)
# gives output

# Predicting a new result
y_pred = regressor.predict(6.5)


# Visualising the Regression results (for higher resolution and smoother curve)
#higher resolution by changing val on first line (smaller = more res)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()