# lecture 72 - DT in R

#copypasta
# Regression Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling - should not be needed in DT b/c DT n/ based on Euclildian distance
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the DT Regression Model to the dataset
install.packages('rpart')
library(rpart)

# needed to add control parameter to get more than one split
# minsplit = 1 gives about four splits, another trap ... -- see notes under 1st ggplot
# trap is resolution is n/ high enough, connecting lines are artefacts 
# (DT is not continuous -- need to change visualization)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1)
                  )   

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Regression Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (DT Regression Model)') +
  xlab('Level') +
  ylab('Salary')

# straight horizontal line comes out similar to SVR in Py (there neeeded FS)
# problem = not enough conditions / splits (no splits = all 1 prediction)

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (DT Regression Model)') +
  xlab('Level') +
  ylab('Salary')