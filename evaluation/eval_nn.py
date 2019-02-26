import sys
sys.path.insert(0, '/training')
from training import train_nn as nn

classifier = nn.classifier()
vect = nn.vectorizer()
