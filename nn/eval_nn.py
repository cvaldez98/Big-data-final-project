import sys
sys.path.append('../')
from train_nn import classifier
from train_nn import vectorizer 
from collection.data_collection_eval import extract_features

import sys
sys.path.append("../")
from collection.data_collection_training import labels
from collection.data_collection_training import all_reviews

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


eval_review_tokens, eval_labels = extract_features()

mlp_count = classifier()
count_vect = vectorizer()

## Score the accuracy of the evaluation set
count_score = mlp_count.score(count_vect.transform(eval_review_tokens), eval_labels)
print("NN MLP with a CountVectorizer performed with an accuracy of " + str(count_score * 100) + " %")

## Predict labels of evaluation set
y_pred_count = mlp_count.predict(count_vect.transform(eval_review_tokens))

## Create confusion matrix
count_matrix = confusion_matrix(eval_labels, y_pred_count)
print(count_matrix)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    

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
plot_confusion_matrix(count_matrix, classes=["neg", "pos"], normalize=True,
                      title='Normalized mlp_count confusion matrix')

plt.show()
