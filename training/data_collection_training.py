import pandas as pd
import numpy as np
import string
import re
import os
from IPython.display import HTML

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import PCA

# from tensorflow.python.keras.models import Sequential, load_model
# from tensorflow.python.keras.layers import Dense, Dropout
# from tensorflow.python.keras import optimizers

import nltk

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import words
from nltk.corpus import wordnet
allEnglishWords = words.words() + [w for w in wordnet.words()]
allEnglishWords = np.unique([x.lower() for x in allEnglishWords])

import plotly.offline as py
import plotly.graph_objs as go
# py.init_notebook_mode(connected=True)

import warnings

# Our imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import sklearn.svm as svm
from imblearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

# data import

path = "../data/aclImdb/"
positiveFiles = [x for x in os.listdir(path+"train/pos/") if x.endswith(".txt")]
negativeFiles = [x for x in os.listdir(path+"train/neg/") if x.endswith(".txt")]
testFiles = [x for x in os.listdir(path+"test/") if x.endswith(".txt")]

# removes punctuation/breaks/etc from a single review
def clean_review(raw_review):
        raw_review = raw_review.replace('<br />', " ")
        raw_review = raw_review.replace('.', " ")
        raw_review = raw_review.replace('-', " ")
        # Remove punctuation, lowercase entire string, split with the (" ") delimiter, remove empty strings
        #  remove punctuation
        raw_review = raw_review.translate(raw_review.maketrans('','',string.punctuation))
        raw_review = raw_review.lower()
        return raw_review

#iterate through the reviews, and put them in a data structure
positiveReviews, negativeReviews, testReviews = [], [], []

for pfile in positiveFiles:
    with open(path+"train/pos/"+pfile, encoding="latin1") as f:
        review = clean_review(f.read())
        positiveReviews.append(review)
for nfile in negativeFiles:
    with open(path+"train/neg/"+nfile, encoding="latin1") as f:
        review = clean_review(f.read())
        negativeReviews.append(review)
for tfile in testFiles:
    with open(path+"test/"+tfile, encoding="latin1") as f:
        review = clean_review(f.read())
        testReviews.append(review)

# print("test reviews: " + str(testReviews))

# merge everything into one
reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "label":1, "file":positiveFiles}),
    pd.DataFrame({"review":negativeReviews, "label":0, "file":negativeFiles}),
    pd.DataFrame({"review":testReviews, "label":-1, "file":testFiles})
], ignore_index=True).sample(frac=1, random_state=1)
# reviews.head()

reviews = reviews[["review", "label", "file"]].sample(frac=1, random_state=1)
train = reviews[reviews.label!=-1].sample(frac=0.6, random_state=1)
valid = reviews[reviews.label!=-1].drop(train.index)
test = reviews[reviews.label==-1]

pos_review_tokens = [indiv_review.split() for indiv_review in positiveReviews]
labels_list = []

for review in positiveReviews:
        labels_list.append('pos')
for review in negativeReviews:
        labels_list.append('neg')

neg_review_tokens = [indiv_review.split() for indiv_review in negativeReviews]
review_tokens = pos_review_tokens + neg_review_tokens

# returns tokens of all reviews
def all_tokens():
        return review_tokens

# returns the labels_list for pos and neg reviews
def labels():
        return labels_list

def all_reviews():
        return positiveReviews + negativeReviews
# print("review_tokens", str(review_tokens))
# make labels array with the same number of posReviews and

# print(train.shape)
# print(valid.shape)
# print(test.shape)

# review_tokens = [indiv_review for indiv_review in train[0]]
# vectorizer = CountVectorizer()
# vectorizer.fit(review_tokens)

# lsvm = LinearSVC()
# lsvm = make_pipeline(sm, lsvm)
# x_train_res, y_train_res = sm.fit_sample(vectorizer.transform(x_train), y_train)

# lsvm.fit(x_train_res, y_train_res)

# get accuracy/performance of classifier
# score = lsvm.score(onehot_enc.transform(x_test), y_test)

# print("SVM Classifier score: the classifier performed on the test set with an accuracy of " + str(score * 100) + " %")

# y_pred = lsvm.predict(onehot_enc.transform(x_test))

