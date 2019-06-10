# Natural Language Processing

# L202 Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)
# quote parameter ignores quotes
# stringsAsFactors false treats strings as strings (not asfactor variables)

# L 203 - goal of creating ind vars in BOW model
# 1 row per review, columns = words in review
# Cleaning the texts - 1st create corpus, then use functions to clean
install.packages('tm') #installin-- text mining package
install.packages('SnowballC') #installing--- word stemming algorithm
library(tm)
# got error " no tm package"
# can use as.character(corpus[[#]]) to see

library(SnowballC)
#also needed to use tools / install package again
corpus = VCorpus(VectorSource(dataset_original$Review))
# L 204
corpus = tm_map(corpus, content_transformer(tolower))
# L 205 - don't want columns for punctuation, numbers, etc.
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
# L 207 still simplifying --- snowballC has the stopwords library
corpus = tm_map(corpus, removeWords, stopwords())
# L 208 stemming
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# L 210
# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
# dtm function creates huge sparse matrix
dtm = removeSparseTerms(dtm, 0.999)
# specify proportion of non-frequent words to be removed, here keeping 99.9% of most freq
# L 211
dataset = as.data.frame(as.matrix(dtm))
# transform dtm into data frame for random forest usage
# data frae is broader than matrix (all values in matrix must be same data type)
dataset$Liked = dataset_original$Liked


# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))
# this gives you liked as dep var vector

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools) # don't forget to install
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
# split ratio is % of sample into test set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# do not need feature scaling (removed)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest) # did you install it?
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)