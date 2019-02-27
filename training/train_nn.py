# use neural networks
from data_collection_training import labels
from data_collection_training import all_reviews

import string
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import optimizers

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
count_vect = CountVectorizer()
hash_vect = HashingVectorizer()
lstm_vect = CountVectorizer()
# tokens must be a list of full reviews
tf_vect.fit(review_tokens)
count_vect.fit(review_tokens)
hash_vect.fit(review_tokens)
lstm_vect.fit(review_tokens)


# split data into training and test set with train_test_split function - setting shuffle to True because, we
# have pos reviews on the 1st half and neg on 2nd half of 'tokens' array

x_train, x_test, y_train, y_test = train_test_split(review_tokens, labels, test_size=.5, random_state=1234, shuffle=True)

# create mlp classifier and train it
# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1)
mlp_count = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1)
# mlp_hash = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1)

# x_train_vect = tf_vect.transform(x_train)
# x_test_vect = tf_vect.transform(x_test)

x_train_vect_count = count_vect.transform(x_train)
x_test_vect_count = count_vect.transform(x_test)

# x_train_vect_hash = hash_vect.transform(x_train)
# x_test_vect_hash = hash_vect.transform(x_test)
# scaling data
# scaler = StandardScaler(with_mean=False)
scaler_count = StandardScaler(with_mean=False)
# scaler_hash = StandardScaler(with_mean=False)

# scaler.fit(x_train_vect)
scaler_count.fit(x_train_vect_count)
# scaler_hash.fit(x_train_vect_hash)

# x_train_vect = scaler.transform(x_train_vect)
x_train_vect_count = scaler_count.transform(x_train_vect_count)
# x_train_vect_hash = scaler_hash.transform(x_train_vect_hash)

# x_test_vect = scaler.transform(x_test_vect)
x_test_vect_count = scaler_count.transform(x_test_vect_count)
# x_test_vect_hash = scaler_hash.transform(x_test_vect_hash)

# mlp.fit(x_train_vect, y_train)
mlp_count.fit(x_train_vect_count, y_train)
# mlp_hash.fit(x_train_vect_hash, y_train)

# get accuracy/performance of classifier
# mlp_score = mlp.score(x_test_vect, y_test)
mlp_count_score = mlp_count.score(x_test_vect_count, y_test)
# mlp_hash_score = mlp_hash.score(x_test_vect_hash, y_test)

# print("MLP with a TfidfVectorizer performed with an accuracy of " + str(mlp_score * 100) + " %")
print("MLP with a count performed with an accuracy of " + str(mlp_count_score * 100) + " %")
# print("MLP with a hash performed with an accuracy of " + str(mlp_hash_score * 100) + " %")

# y_pred_count = count_lsvm.predict(count_vect.transform(x_test))
# y_pred_tf = mlp.predict(x_test_vect)
y_pred_count = mlp_count.predict(x_test_vect_count)
# y_pred_hash = mlp_hash.predict(x_test_vect_hash)
# print(y_pred)
bin_labels = ['pos', 'neg']
# recall_score = recall_score(y_test, y_pred_tf, labels=bin_labels, average=None)
# precision_score = precision_score(y_test, y_pred_tf, labels=bin_labels, average=None)
# print("The recall score with tf is " + str(recall_score))
# print("The percision score with tf is " + str(precision_score))

recall_score_2 = recall_score(y_test, y_pred_count, labels=bin_labels, average=None)
precision_score_2 = precision_score(y_test, y_pred_count, labels=bin_labels, average=None)
print("The recall score with count is " + str(recall_score_2))
print("The percision score with count is " + str(precision_score_2))

# recall_score_3 = recall_score(y_test, y_pred_hash, labels=bin_labels, average=None)
# precision_score_3 = precision_score(y_test, y_pred_hash, labels=bin_labels, average=None)
# print("The recall score with hashing is " + str(recall_score_3))
# print("The percision score with hashing is " + str(precision_score_3))

############### Trying LSTM and Keras (sequential) ##############################
max_features = 1024

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


x_train_lstm = lstm_vect.transform(x_train)
x_test_lstm = lstm_vect.transform(x_test)

model.fit(x_train_lstm, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test_lstm, y_test, batch_size=16)

print("lstm score: " + str(score))

def classifier():
    return mlp_count

def vectorizer():
    return tf_vect
