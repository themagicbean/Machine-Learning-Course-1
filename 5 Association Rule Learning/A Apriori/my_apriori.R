# Apriori

# Data Preprocessing

# install.packages('arules')
library(arules) # may need to install first (above line)

dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE) # header = false means no titles of columns
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
#second line gets sparse matrix, specify separator (can view in text ed), rm.duplicates means there are duplicates in data
#gives response with duplicates

summary(dataset)
itemFrequencyPlot(dataset, topN = 10) #args are dataset and # of most sold product

# L 159 selecting support and confidence values

# Training Apriori on the dataset (via arules package)
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))
# change support and confidence values to get workable # of rules

# L 160
# Visualising the results 
inspect(sort(rules, by = 'lift')[1:10])