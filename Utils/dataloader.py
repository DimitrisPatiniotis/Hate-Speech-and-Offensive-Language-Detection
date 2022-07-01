from numpy import concatenate
from settings import *
from general_utils import *
from time import time
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub


class DataLoader():
    def __init__(self, stemming=True, data_csv_path=DATA_CSV, clean_tweets=True, train_test_split = 0.25, n_gram=1):
        self.data_csv_path = data_csv_path
        self.stemming = stemming
        self.clean_tweets = clean_tweets
        self.train_test_split = train_test_split
        self.n_gram = n_gram
        self.tokenizer = None
        self.data = None
        self.tweets = None
        self.X_train_unchanged = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def undersample(self):
        remove_1 = 19190-1430
        remove_2 = 4163-1430
        neither = self.tweets.loc[self.tweets['class'] ==  0]
        hate_speach = self.tweets.loc[self.tweets['class'] ==  1]
        offensive_language = self.tweets.loc[self.tweets['class'] ==  2]
        hate_drop_indx = np.random.choice(hate_speach.index, remove_1, replace=False)
        offensive_drop_indx = np.random.choice(offensive_language.index, remove_2, replace=False)
        hate_under = hate_speach.drop(hate_drop_indx)
        offensive_under = offensive_language.drop(offensive_drop_indx)
        self.tweets = pd.concat([neither, hate_under, offensive_under])

    def load(self, print_details=False):
        # load
        self.tweets = load_data(print_info=print_details)
        self.undersample()
        # clean
        if self.clean_tweets:
            if print_details:
                print('Cleaning tweets...')
                starting_time = time()
            self.tweets['tweet'] = self.tweets['tweet'].map(lambda x: tweet_cleaner(str(x), stemming=self.stemming))
            if print_details:
                ending_time = time()
                print('{} tweets cleaned in {} seconds...'.format(len(self.tweets['tweet']), round(ending_time - starting_time, 2)))


    def split_data(self):
        x = self.tweets['tweet']
        y = self.tweets['class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size = self.train_test_split, random_state=42, stratify=y)
        self.X_train_unchanged = self.X_train

    def bag_of_words(self):
        vectorizer = CountVectorizer(analyzer="word", ngram_range=(self.n_gram,self.n_gram))
        self.X_train = vectorizer.fit_transform(self.X_train)
        self.X_test = vectorizer.transform(self.X_test)

    def prepare_for_ml_models(self, plot=False):
        self.load()
        self.split_data()
        if plot:
            self.get_len_distr()
        self.bag_of_words()
    
    def prepare_for_lstm(self):
        self.load()
        self.split_data()

    def tokenize_and_pad(self):
        max_words = 30000
        max_len = 128
        self.tokenizer = Tokenizer(num_words=max_words)
        self.tokenizer.fit_on_texts(self.X_train)
        self.X_train = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_train = pad_sequences(self.X_train,maxlen=max_len)
        self.X_test = self.tokenizer.texts_to_sequences(self.X_test)
        self.X_test = pad_sequences(self.X_test, maxlen=max_len)
        return 

    def get_tweet_bert_embeding(tweet):
        bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
        preprocessed_text = bert_preprocess(tweet)
        return bert_encoder(preprocessed_text)['pooled_output']

    # PLOTS
    def get_len_distr(self):
        frequency_distribution = FreqDist(word_tokenize(' '.join(self.tweets['tweet'])))
        frequency_distribution.plot(30,cumulative=False)
        plt.show()
    

if __name__ == '__main__':
    print('Data Loader Util')