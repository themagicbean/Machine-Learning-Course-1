# data preprocessing
# set wd -- use files menu
# click more, click set as wd

#import dataset
dataset = read.csv('Data.csv')

#indexes START AT ONE !!!!
# no need to do matrix of features / dep var like py??

#lecture 13 / missing data
# need to do each missing data individually

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

#is.na checks returns true if value missing
# FUN = function R syntax ?? , mean function exists in R
# na.rm = true (include missing values when computing mean)

#redo for Salary ILO Age
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# vid 14 "simpler" no need for hot encoding
# encoding categorical data
# transform country column into column of factors
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))
#f1 on factor gives info
# levels gives categories, labels is NON-ORDINAL 

dataset$Purchased = factor(dataset$Purchased,
                          levels = c('No', 'Yes'),
                          labels = c(0, 1))

# lecture 16 splitting into training and test sets
# install lib 
# install.packages('caTools')
# above line was ran then changed to # b/c no need to repeat

library(caTools)
# or you can click on the box in the packages tab


#to set same random seed for same result cross machine
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# splits test/train
# note arguments are different (no X) and 0.8 = 20% split from py
# returns T or F and then we assign
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
# to view, click on items in "environment" pane

# lecture 17 feature scaling - these lines generate errors, see notes
# training_set = scale(training_Set)
# test_set = scale(test_set)
# holy shit that was easy
# but leads to error "X must be numeric" 

# country and purchase, though appear numeric, have been factorized
# and therefore are not actually numeric
# need to exclude categorias from scaling
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])
# specify columns 2-3 as those are the only ones to scale
# this gets error for training set only ???? ... nvmind seems fine now lol

# most libraries d/n require feature scaling but should know it

# for some reason files look fine in NP++ but open blank here?
