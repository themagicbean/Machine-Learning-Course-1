# lecture 60 polynomial regresion in R

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3] # keeps cols 2 & 3 (R numbers from 1)


# Splitting the dataset into the Training set and Test set - NOT DONE TOO SMALL SAMPLE
# install.packages('caTools')
"""library(caTools)
set.seed(123)
split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)"""

# Feature Scaling -- ALSO NOT NEEDED
# training_set = scale(training_set)
# test_set = scale(test_set)

# L61
# fit linreg
lin_reg = lm(formula = Salary ~ .,
             data = dataset)
# after run, "summary(lin_reg)" in console gives info 

#fit poly reg
# add poly variables by adding columns 
dataset$Level2 = dataset$Level^2 # adds a new column w /squared var
dataset$Level3 = dataset$Level^3 # ditto .. couldn't this be a for loop?
dataset$Level4 = dataset$Level^4 # added at end of L62
poly_reg = lm(formula = Salary ~ .,  # need to refit after each change to dataset
              data = dataset)
#again best to use summary to view stats.  "." includes all columns

#L62 visualising results

# visualising LR
# install.packages('ggplot2') --- if not already installed
library(ggplot2)  # selects the lib
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (LR)') +
  xlab('Level') +
  ylab('Celery')

# visualising PR - change part of line from lin_reg to poly_reg
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (PR)') +
  xlab('Level') +
  ylab('Celery')

#L63 
# predicting a new result with LR - making a single new prediction @ 6.5 level
y_pred = predict(lin_reg, data.frame(Level = 6.5))
# then can type y_pred in console to get value (or is in values in environment pane)

# predicting with PR / single pred at level = .  need to add in other levels
y_pred2 = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))

# lecture 64 gets into R regression template








