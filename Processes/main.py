import sys
sys.path.insert(1, '../Utils/')
from models import naive_bayes_model, logistic_regression_model, bert_lstm_model, svm_model, random_forest_model, adaboost_model, LSTM_model, bert_model
from dataloader import DataLoader
from settings import *




def run_ml():
    loader = DataLoader()
    loader.prepare_for_ml_models()
    naive_bayes_model(loader)
    logistic_regression_model(loader)
    svm_model(loader)
    random_forest_model(loader)
    adaboost_model(loader)

def run_lstm():
    loader = DataLoader()
    loader.load()
    loader.split_data()
    loader.tokenize_and_pad()
    LSTM_model(loader)


def run_bert(lstm=True):
    loader = DataLoader()
    loader.load()
    loader.split_data()
    if lstm:
        bert_lstm_model(loader)
    else:
        bert_model(loader)

def main():
    run_ml()
    run_lstm()
    run_bert(lstm=False)
    run_bert()


if __name__ == '__main__':
    main()