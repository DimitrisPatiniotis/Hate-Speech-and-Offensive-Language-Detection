from settings import *
from matplotlib import pyplot as plt

# ML Imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,ConfusionMatrixDisplay

# LSTM imports
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
import tensorflow as tf 

from general_utils import test_data_results
import tensorflow_text as text
# Bert
import tensorflow_hub as hub


def plot_acc(history, model_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('{} Model Accuracy'.format(model_name))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def naive_bayes_model(dataloader, print_report=True):
    model = MultinomialNB()
    model.fit(dataloader.X_train, dataloader.y_train)
    pred = model.predict(dataloader.X_test)
    accuracy = round(accuracy_score(dataloader.y_test,pred),2)
    if print_report:
        print('Naive Bayes Model Report:')
        print(classification_report(dataloader.y_test,pred,target_names=[num_to_class[i] for i in range(3)]))

    cm = confusion_matrix(dataloader.y_test, pred, labels= model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=num_to_class.values())
    display.plot()
    plt.title('Naive Bayes Confusion Matrix')
    plt.show()
    return model

def logistic_regression_model(dataloader, print_report=True):
    model = LogisticRegression(random_state=0, max_iter=500)
    model.fit(dataloader.X_train, dataloader.y_train)
    pred = model.predict(dataloader.X_test)
    accuracy = round(accuracy_score(dataloader.y_test,pred),2)
    if print_report:
        print('Logistic Regression Report:')
        print(classification_report(dataloader.y_test,pred,target_names=[num_to_class[i] for i in range(3)]))
    return model

def svm_model(dataloader, print_report=True):
    model = SVC()
    model.fit(dataloader.X_train, dataloader.y_train)
    pred = model.predict(dataloader.X_test)
    if print_report:
        print('SVM Report:')
        print(classification_report(dataloader.y_test,pred,target_names=[num_to_class[i] for i in range(3)]))
    return model

def random_forest_model(dataloader, print_report=True):
    model = RandomForestClassifier(max_depth=5, random_state=0)
    model.fit(dataloader.X_train, dataloader.y_train)
    pred = model.predict(dataloader.X_test)
    accuracy = round(accuracy_score(dataloader.y_test,pred),2)
    if print_report:
        print('Random Forest Report:')
        print(classification_report(dataloader.y_test,pred,target_names=[num_to_class[i] for i in range(3)]))

    cm = confusion_matrix(dataloader.y_test, pred, labels= model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=num_to_class.values())
    display.plot()
    plt.title('RF Confusion Matrix')
    plt.show()
    return model
    
def adaboost_model(dataloader, print_report=True):
    model = AdaBoostClassifier(n_estimators=100, random_state=0)
    model.fit(dataloader.X_train, dataloader.y_train)
    pred = model.predict(dataloader.X_test)
    accuracy = round(accuracy_score(dataloader.y_test,pred),2)
    if print_report:
        print('AdaBoost Report:')
        print(classification_report(dataloader.y_test,pred,target_names=[num_to_class[i] for i in range(3)]))
    return model

def LSTM_model(dataloader, plot=False):
    model = Sequential()
    model.add(Embedding(25000, 100, input_length=128))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='sigmoid'))
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history =  model.fit(dataloader.X_train, dataloader.y_train, validation_split=0.1, batch_size=128, epochs=20)
    if plot:
        plot_acc(history, 'LSTM')
    print('\nLSTM classification report:')
    test_data_results(model, dataloader)
    return model, history

def bert_model(dataloader, plot=False):
    bert_preprocess = hub.KerasLayer("../3")
    bert_encoder = hub.KerasLayer("../4")
    # Bert Layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    # Neural network layers
    l = tf.keras.layers.Dropout(0.2, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(3, activation='softmax', name="output")(l)
    # Use inputs and outputs to construct a final model
    model = tf.keras.Model(inputs=[text_input], outputs = [l])
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(dataloader.X_train, dataloader.y_train, validation_split=0.1, epochs=20)
    if plot:
        plot_acc(history, 'BERT')
    print('\nBERT classification report:')
    test_data_results(model, dataloader)
    return model

def bert_lstm_model(dataloader, plot=False):
    bert_preprocess = hub.KerasLayer("../3")
    bert_encoder = hub.KerasLayer("../4")
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    l = tf.keras.layers.Dropout(0.2, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Reshape((1,768))(l)
    l = tf.keras.layers.LSTM(128, dropout=0.2, input_shape=(128,1,768))(l)
    l = tf.keras.layers.Dense(3, activation='softmax', name="output")(l)
    model = tf.keras.Model(inputs=[text_input], outputs = [l])
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(dataloader.X_train, dataloader.y_train, validation_split=0.1, epochs=20)
    if plot:
        plot_acc(history, 'Bert + LSTM')
    return model
