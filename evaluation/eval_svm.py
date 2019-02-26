import sys
sys.path.insert(0, '/training')
from training import train_svm as svm

classifier = svm.classifier()
vect = svm.vectorizer()