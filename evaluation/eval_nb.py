import sys
sys.path.insert(0, '/training')
from training import train_nb as nb

classifier = nb.classifier()
vect = nb.vectorizer()