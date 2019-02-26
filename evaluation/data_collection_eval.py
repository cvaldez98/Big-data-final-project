# cross validation here

import os
import string
# give a result from the million + reviews

directory = os.fsencode('data/movie_samples/')
str_directory  = 'data/movie_samples/'
REVIEW_PREFIX = 'review/text'
reviews = []     

def evaluate_snippet(filename):
    print("evaluating",filename)
    # evaluate review
    # here we should use our trained classifier and run it here.
    # maybe we should also print our statistics here?
    with open('data/movie_samples/' + filename, 'r', encoding='latin-1') as f:
        for _, line in enumerate(f):
                if REVIEW_PREFIX in line:
                        line = line.replace("<br />", " ")
                        line = line.translate(line.maketrans(' ',' ',string.punctuation))
                        line = line.lower()
                        reviews.append(line[len(REVIEW_PREFIX):])


# a list of around one million reviews
def review_tokens():
        return reviews


for file in os.listdir(str_directory):
    filename = os.fsdecode(file)
    if(filename.endswith('.txt')):
        evaluate_snippet(filename)
        continue


