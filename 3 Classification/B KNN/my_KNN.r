# L 100 KNN

#copypasta 

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting K-NN to the Training set !!! --- !!!
# and predicting results, in one step
# install.packages('class')  
library(class)
y_pred = knn(train = training_set[, -3], 
             test = test_set[, -3],
             cl = training_set[, 3],
             k = 5)
# need to exclude dep var from data because it is to be predicted
# cl = factor of true classifications (actual results to use for prediction?)
# can view y_pred in console

# removed next line re:
# Predicting the Test set results, was done in one line


# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
# !!! --- need to change display template --- !!!
# because no longer a "classifier" - not created separately from y_pred

library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#y_grid = predict(classifier, newdata = grid_set) -- removed 
y_grid = knn(train = training_set[, -3], 
             test = grid_set[, -3],  # change test to grid set to predict imaginary pixels
             cl = training_set[, 3],
             k = 5)
#above is new/changed  line
plot(set[, -3],
     main = 'K-NN (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#y_grid = predict(classifier, newdata = grid_set) -- removed 
y_grid = knn(train = training_set[, -3], 
             test = grid_set[, -3],  # change test to grid set to predict imaginary pixels
             cl = training_set[, 3],
             k = 5)
# above is changed line
plot(set[, -3], main = 'K-NN (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))