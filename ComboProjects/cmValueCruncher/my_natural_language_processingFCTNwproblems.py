# Natural Language Processing
# L191
#making model to predict if review positive or negative
def NLPNB():
# Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Importing the dataset
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    # read csv can also read tsv files!
    # but need to specify delimiter (it's not a comma) 
    # quoting parameter 3 = to ignore double quotes
    
    # L 192
    # Cleaning the texts
    import re
    import nltk
    nltk.download('stopwords')
    # list of filler / minor words like is / yours / so etc., for removal
    # but includes not & sim!  so can reverse tone of some samples
    from nltk.corpus import stopwords
    # gotta both download and import stopwords list
    from nltk.stem.porter import PorterStemmer
    # trims words to stems (removes endings)
    corpus = [] # L 197, this is the list of cleaned-up reviews
    for i in range(0, 1000): #basics in 193-94, loop in 195-96-97
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        # removes all non-letter characters
        # first paramter, quotes in bracket w/ hats are what you don't want to remove
        # second paramter ensures removed characters are replaced with spaces
            # so words don't stick together
        # third parameter is on what to apply the rule/remover on (dataset)
        
        review = review.lower()
        # L193, changes all letters to lowercase
        review = review.split()
        # L 194, makes review a list of its words (each word an element)
        
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        # for loop to apply ps (stemming) to non-stopwords words in review (L194 and 195)
        review = ' '.join(review)
        # L 196 rejoining elements of cleaned review into single string
        corpus.append(review)
        # adding single string to corpus
    
    # L198 Creating the Bag of Words model (see notes in notepad)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    # max features reduces sparsity 
    X = cv.fit_transform(corpus).toarray()
    # creates huge sparse matrix (matrix of features)
    # L199 trying to reduce sparsity
    y = dataset.iloc[:, 1].values
    # .iloc takes columns when importing from pandas
    # : takes all reviews
    # .vallues creates values
    # this is dep var
    # also look to part 9 dimensionality reduction techniques
    
    # 200
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]