# lecture 28 et seq
# set wd -- find folder in files menu, click more, set wd

dataset = read.csv('Salary_Data.csv')

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
# split 2/3 gives 2/3 to training (opposite of py)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
#d/n need featurescaling oh yeah woo hoo still waiting for next vid to load

#lecture 29
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)
#f1 to check fn, args for lm(fitting linear model)

# !!!
#type "summary(regressor)" in console to get info re: model
# !!!

# stars to right margin indicate statistical significance (0-3 stars)
# p value want < 5% ... less is better

# lecture 30 
y_pred = predict(regressor, newdata = test_set)
# type y_pred in console to see vals

# lecture 31  visualising results
# need ggplot2 lib (following line commented after run !!!)
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Slavery') +
  ylab('PaTHEtic COMPensATION')
# watch syntax, use + on 1st line

# lecture 31 now try the test set
# would need ggplot in the library (library(ggplot2)) if hadn't already
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  # for points, only change test set in first line to replace training data w test data
  # for line, do not need to replace bc training is base of model, regressor already trained
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Testy McSetsAlot)') +
  xlab('Years of Slavery') +
  ylab('PaTHEtic COMPensATION')
# execution gets a new graph (red points new)


