# Artificial Neural Network

"""
installing Keras
And Windows users, please open the anaconda prompt, which you can find this way:
Windows Button in the lower left corner -> List of programs -> anaconda -> anaconda prompt
Then inside the anaconda prompt, copy-paste and enter the following line command:
    """
# gives warning re: deprecated conda.compat module
# updated conda, superseded certifi -> conda-forge
    
""" at end of install:
(base) C:\Users\Admin>ET _sysp=%~dpA
'ET' is not recognized as an internal or external command,
operable program or batch file.

(base) C:\Users\Admin>IF NOT EXIST "!_sysp!\Scripts\conda.exe"
"""
# then it did more ...



"""outdated version - use above
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

"""
C:\Users\Admin\Anaconda3\lib\site-packages\sklearn\preprocessing\_encoders.py:371: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
If you want the future behaviour and silence this warning, you can specify "categories='auto'".
In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
  warnings.warn(msg, FutureWarning)
C:\Users\Admin\Anaconda3\lib\site-packages\sklearn\preprocessing\_encoders.py:392: DeprecationWarning: The 'categorical_features' keyword is deprecated in version 0.20 and will be removed in 0.22. You can use the ColumnTransformer instead.
  "use the ColumnTransformer instead.", DeprecationWarning)
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# so far so good




# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #initializes
from keras.layers import Dense # builds layers


#next tutorial, L230 "step 4"
# Initialising the ANN - defining it as a sequence as layers
# (alternatively,ANN initialization can be by defining a graph)

classifier = Sequential() #"classifier" is the ANN we are building, an objecct of seq'l class

# L 229 "step 5 .. these are short steps ..."

# Adding the input layer and the first hidden layer
# dense fn randomizes weights of init nodes near, but not, zero
# use generally rectifier function, sigmoid for output (t/f)
# cf training ANN slide on intro

# add method adds layers

#input layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#output_dim = number of nodes in layer we are adding -- important issue -- 
#           cannot predict ideal # of nodes ... here, output = 1, input = 1, avg = 6
#           in hidden layer, general starting point is avg. of # of input & output nodes
#           to do better, study parameter tuning, including k-fold techniques (later)
# init = uniform makes random weights =, small
# activation = relu = rectifier  *** see L 244 on ReLU layer *** (REctified Linear Unit layer) 
#input_dim = # of ind. vars (needed in init layer)
# more deprecation warnings
"""
serWarning: Update your `Dense` call to the Keras 2 API:
    `Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")`
"""


# L 232 adding layers ...
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# L233
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# using sigmoid b/c binary(boolean) output
# if >2 dep vars, output_dim = # of dep vars, activation = softmax

# L 234
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# compiling = applyting s. grad. desc. on entire network
# optimizer = algorithm for optimization, several types of SGD, adam is 'efficient'
# loss = loss fn in SGD (loss fn must be optimized, sim to logistic regression kinda)
#   but a little different - logarithimic loss in sigmoid fn = 'binary_crossentropy"
#   if 3d or more 'categoric(al?)_crossentropy
# metrics = criterium to analyze ... accuracy, can use list of metrics [] 

# L 235 - can watch fitting w output metric (accuracy) per run
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# first arg is train set, second is actual outcomes
# batch size: # of observations after which change weights (no rules ... experiment)
# another deprec warning, nb_epoch in fit renamed "epochs" **
# takes a while ....
# video had 85% return at max?  differences in module?

#L236 confusion matrix, model eval
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# converts probabilities into binary/boolean value via 0.5 threshold 
# can use higher threshold if, e.g., medicine (predicting malignant tumor)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# compute accuracy in console ... mine slightly < his 
# correct / total = accuracy ... have 84% 
# w/o parameter tuning
# similar to training set so good