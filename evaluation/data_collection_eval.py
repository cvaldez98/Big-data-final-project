# cross validation here

import os

# give a result from the million + reviews

directory = os.fsencode('../data/movie_samples/')

def evaluate_snippet(filename):
    print("evaluating",filename)
    # evaluate review
    # here we should use our trained classifier and run it here.
    # maybe we should also print our statistics here?


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if(filename.endswith('.txt')):
        evaluate_snippet(filename)
        continue


