import sys
sys.path.append('../')
from train_svm import classifier
from train_svm import vectorizer 
from data_collection.data_collection_eval import extract_features


import string
import numpy as np
import pandas as pd 
import sklearn.svm as svm
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing  import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

import plotly
import plotly.offline as py
import plotly.graph_objs as go
#plotly.offline.init_notebook_mode(connected=True)

import itertools
import matplotlib.pyplot as plt

eval_review_tokens, eval_labels = extract_features()

tf_lsvm = classifier()
tf_vect = vectorizer()

## Score the accuracy of the evaluation set
tf_score = tf_lsvm.score(tf_vect.transform(eval_review_tokens), eval_labels)
print("SVM with a TfidfVectorizer performed with an accuracy of " + str(tf_score * 100) + " %")

## Predict labels of evaluation set
y_pred_tf = tf_lsvm.predict(tf_vect.transform(eval_review_tokens))

## Create confusion matrix
tf_matrix = confusion_matrix(eval_labels, y_pred_tf)

# print(cnf_matrix)
print(tf_matrix)
# print(hash_matrix)



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
# plot_confusion_matrix(cnf_matrix, classes=["neg", "pos"], normalize=True,
#                       title='Normalized count_vect confusion matrix')
plot_confusion_matrix(tf_matrix, classes=["neg", "pos"], normalize=True,
                      title='Normalized tf_vect confusion matrix')
# plot_confusion_matrix(hash_matrix, classes=["neg", "pos"], normalize=True,
#                       title='Normalized hash_vect confusion matrix')

plt.show()
