from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Activation, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():  # load data excel dengan pandas
    df = pd.read_csv('dataset.csv', delimiter='\t')
    return df


def tokenize_tweet(data_frame):  # tokenize untuk mendapatkan text sequence dan index word
    text = data_frame["Tweet"].tolist()
    token = Tokenizer()
    token.fit_on_texts(text)
    token = {'sqnce': token.texts_to_sequences(
        text), 'vocab': len(token.index_word)+1}
    return token


def sequences_tweet(encode_text, max_kata):
    x = pad_sequences(encode_text, maxlen=max_kata, padding="post")
    return x


def train_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=40, test_size=0.3, stratify=y)

    train = {
        'x_train': np.asarray(x_train),
        'y_train': np.asarray(y_train),
        'x_test': np.asarray(x_test),
        'y_test': np.asarray(y_test)
    }
    return train


def model(vocab, vec_size, max_kata, x_train, y_train, x_test, y_test, ex):
    model = Sequential()
    model.add(Embedding(vocab, vec_size, input_length=max_kata))
    model.add(Conv1D(64, 8, activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    return model.predict_classes(ex)


def get_encode(data_frame, x, max_kata):
    text = data_frame["Tweet"].tolist()
    token = Tokenizer()
    token.fit_on_texts(text)
    x = token.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=max_kata, padding="post")
    return x


data = load_data()  # load data

token = tokenize_tweet(data)
sqnce = token['sqnce']  # sequence
vocab = token['vocab']  # vocab

max_kata = 100
x = sequences_tweet(sqnce, max_kata)  # pad sequences
y = to_categorical(data["sentimen"])  # y categorical

train = train_dataset(x, y)  # train

ex = ['hari ini ibu memberi hadiah jadi aku senang']
ex = get_encode(data, ex, max_kata)

vec_size = 300
predict = model(vocab, vec_size, max_kata,
                train['x_train'], train['y_train'], train['x_test'], train['y_test'], ex)

print(predict)
