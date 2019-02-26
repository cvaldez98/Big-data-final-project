# use neural networks
from data_collection_training import labels
from data_collection_training import all_reviews

import string
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

import plotly
import plotly.offline as py
import plotly.graph_objs as go
# plotly.offline.init_notebook_mode(connected=True)

import itertools
import matplotlib.pyplot as plt

labels = labels()
review_tokens = all_reviews()

# vectorizer for organizing training/testing data
tf_vect = TfidfVectorizer()
# tokens must be a list of full reviews
tf_vect.fit(review_tokens)

# split data into training and test set with train_test_split function - setting shuffle to True because, we
# have pos reviews on the 1st half and neg on 2nd half of 'tokens' array

x_train, x_test, y_train, y_test = train_test_split(review_tokens, labels, test_size=.5, random_state=1234, shuffle=True)

# create mlp classifier and train it
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1)

x_train_vect = tf_vect.transform(x_train)
x_test_vect = tf_vect.transform(x_test)
# scaling data
scaler = StandardScaler(with_mean=False)
scaler.fit(x_train_vect)
x_train_vect = scaler.transform(x_train_vect)
x_test_vect = scaler.transform(x_test_vect)

mlp.fit(x_train_vect, y_train)

# get accuracy/performance of classifier
mlp_score = mlp.score(x_test_vect, y_test)

print("MLP with a TfidfVectorizer performed with an accuracy of " + str(mlp_score * 100) + " %")

# y_pred_count = count_lsvm.predict(count_vect.transform(x_test))
y_pred_tf = mlp.predict(x_test_vect)
# y_pred_hash = hash_lsvm.predict(hash_vect.transform(x_test))
# print(y_pred)

def classifier():
    return mlp

def vectorizer():
    return tf_vect
