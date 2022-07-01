from time import time
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from settings import *

from sklearn.metrics import classification_report

# Returns str
def tweet_cleaner(tweet, stemming=False):
    # Tokenization
    tweet = word_tokenize(str(tweet).lower().replace("_", " ").replace("#", ""))
    # Remove special characters
    tweet = [i for i in tweet if i.isalpha()]
    # Stemming
    if stemming:
        tweet = " ".join([PorterStemmer().stem(i) for i in tweet if i not in (stopwords.words('english'))])
    # Removing links
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)
    # Removing leading and trailing spaces
    tweet = tweet.strip(' ')

    return tweet

# Returns a pd.Dataframe
def load_data(print_info=True):
    if print_info:
        print('Loading data...')
        starting_time = time()
    tweets = pd.read_csv(DATA_CSV)
    tweets.drop(['Unnamed: 0','count','hate_speech','offensive_language','neither'],axis=1,inplace=True)
    if print_info:
        ending_time = time()
        print('Data loaded in {} seconds.'.format(round(ending_time-starting_time, 4)))
        classes = ['Offensive Language', 'Hate Speech', 'Neither']
        occurances_ls = tweets['class'].value_counts()
        occurances = [occurances_ls[1], occurances_ls[2], occurances_ls[0]]
        plt.figure(figsize = (10, 5))
        plt.bar(classes, occurances, color ='blue', width = 0.4)
        plt.xlabel("Class")
        plt.ylabel("Occurances")
        plt.title("Class Distribution")
        plt.show()
    return tweets

def test_data_results(model, dataloader):
    y_test = dataloader.y_test
    predicitons = model.predict(dataloader.X_test)
    predicitons = np.array([list(i).index(max(i)) for i in predicitons])
    print(classification_report(y_test, predicitons))

if __name__ == '__main__':
    load_data()