# use naive bayes
import sys
sys.path.append("../")
from collection.data_collection_training import labels
from collection.data_collection_training import all_reviews


import string
import numpy as np
from numpy import array
import pandas as pd
import sklearn.svm as svm
from sklearn import datasets
from sklearn.metrics import recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing  import LabelEncoder

from sklearn.model_selection import train_test_split
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
count_vect = CountVectorizer()
# hash_vect = HashingVectorizer()
# tokens must be a list of full reviews
tf_vect.fit(review_tokens)
count_vect.fit(review_tokens)

# split data into training and test set with train_test_split function - setting shuffle to True because, we
# have pos reviews on the 1st half and neg on 2nd half of 'tokens' array

x_train, x_test, y_train, y_test = train_test_split(review_tokens, labels, test_size=.5, random_state=1234, shuffle=True)

# create naive bayes classifier and train it
# count_nb = MultinomialNB()
tf_nb = MultinomialNB()
count_nb = MultinomialNB()
# multinomial - 86.416%
# bernoulli - 85.1%
print("Training data...")
tf_nb.fit(tf_vect.transform(x_train), y_train)
count_nb.fit(count_vect.transform(x_train), y_train)
print("Data trained.")

# get accuracy/performance of classifier
tf_score = tf_nb.score(tf_vect.transform(x_test), y_test)
count_score = count_nb.score(count_vect.transform(x_test), y_test)
print("NB with a TfidfVectorizer performed with an accuracy of " + str(tf_score * 100) + " %")
print("NB with a Count performed with an accuracy of " + str(count_score * 100) + " %")

y_pred_count = count_nb.predict(count_vect.transform(x_test))
y_pred_tf = tf_nb.predict(tf_vect.transform(x_test))
# print(y_pred)
bin_labels = ['pos', 'neg']
recall_score = recall_score(y_test, y_pred_tf, labels=bin_labels, average=None)
precision_score = precision_score(y_test, y_pred_tf, labels=bin_labels, average=None)
print("The recall score with tf is " + str(recall_score))
print("The percision score with tf is " + str(precision_score))

def classifier():
    return tf_nb

def vectorizer():
    return tf_vect


# Compute confusion matrix
tf_matrix = confusion_matrix(y_test, y_pred_tf)
print(tf_matrix)


def plot_confusion_matrix(cm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
     """
     #This function prints and plots the confusion matrix.
     #Normalization can be applied by setting `normalize=True`.
     """
     if normalize:
         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
         print("Normalized confusion matrix")
     else:
         print('Confusion matrix, without normalization')

     print(cm)

     plt.imshow(cm, interpolation='nearest', cmap=cmap)
     plt.title(title)
     plt.colorbar()
     tick_marks = np.arange(len(classes))
     plt.xticks(tick_marks, classes, rotation=45)
     plt.yticks(tick_marks, classes)

     fmt = '.2f' if normalize else 'd'
     thresh = cm.max() / 2.
     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

     plt.ylabel('True label')
     plt.xlabel('Predicted label')
     plt.tight_layout()

     np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
# # plot_confusion_matrix(cnf_matrix, classes=["neg", "pos"], normalize=True,
# #                       title='Normalized count_vect confusion matrix')
plot_confusion_matrix(tf_matrix, classes=["neg", "pos"], normalize=True,
                       title='Normalized tf_vect confusion matrix')
# # plot_confusion_matrix(hash_matrix, classes=["neg", "pos"], normalize=True,
# #                       title='Normalized hash_vect confusion matrix')

plt.show()