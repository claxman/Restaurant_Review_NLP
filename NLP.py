#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing
# 
# Author: Chaitanya Laxman

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# ## Cleaning the texts

# In[32]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    temp = []
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    for word in review:
        if word not in all_stopwords:
            temp.append(ps.stem(word))
    review = ' '.join(temp)
    corpus.append(review)


# ## Creating the Bag of Words model

# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# ## Splitting the dataset into the Training set and Test set

# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ## Training the Naive Bayes model on the Training set

# In[41]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# ## Predicting the Test set results

# In[42]:


y_pred = classifier.predict(X_test)


# ## Making the Confusion Matrix

# In[44]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test,y_pred)

