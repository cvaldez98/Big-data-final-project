# cross validation here

import os
import pickle
import string

# give a result from the million + reviews

directory = os.fsencode('data/movie_samples/')

PICKLE_DIR = "data/pickle/"
EVAL_REVIEWS_PICKLE_PATH = "data/pickle/eval_reviews.pickle"
EVAL_LABELS_PICKLE_PATH = "data/pickle/eval_labels.pickle"
REVIEW_PREFIX = "review/text"
LABEL_PREFIX = "review/score: "

reviews = []
true_labels = []

def clean_data(filename):
        eval_set = open(filename, encoding = "ISO-8859-1")
        for _, line in enumerate(eval_set):
                if REVIEW_PREFIX in line:
                        line = line.replace("<br />", " ")
                        line = line.translate(line.maketrans(' ',' ',string.punctuation))
                        line = line.lower()
                        #print(line)
                        reviews.append(line[len(REVIEW_PREFIX):])
                if LABEL_PREFIX in line:
                        line = float(line[len(LABEL_PREFIX):])
                        #print(line)
                        if (line > 3.0):
                                true_labels.append("pos")
                        else:
                                true_labels.append("neg")
                

        with open(EVAL_REVIEWS_PICKLE_PATH, "wb") as fp:   #Pickling
                pickle.dump(reviews, fp)
        with open(EVAL_LABELS_PICKLE_PATH, "wb") as fp:   #Pickling
                pickle.dump(true_labels, fp)

def evaluate_snippet(filename):
        print("evaluating",filename)
        clean_data(filename)
        # evaluate review
        # here we should use our trained classifier and run it here.
        # maybe we should also print our statistics here?

def main():
        try:
                reviews = pickle.load(open(EVAL_REVIEWS_PICKLE_PATH, "rb"))
                labels = pickle.load(open(EVAL_LABELS_PICKLE_PATH, "rb"))
        except (OSError, IOError) as e:
                if not os.path.exists(PICKLE_DIR):
                        os.makedirs(PICKLE_DIR)
                for file in os.listdir(directory):
                        filename = os.fsdecode(directory+file)
                        if(filename.endswith('.txt')):
                                evaluate_snippet(filename)
                                continue

def get_reviews():
        return reviews
def get_labels():
        return true_labels
        
main()

