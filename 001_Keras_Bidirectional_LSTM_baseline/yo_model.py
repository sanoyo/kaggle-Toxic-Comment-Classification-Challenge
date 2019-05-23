
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pdb


maxlen = 100
max_features = 20000

# train = pd.read_csv("../csv/train.csv")
# test = pd.read_csv("../csv/test.csv")
#
# list_sentences_train = train["comment_text"]
# list_sentences_test = test["comment_text"]


# lists = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#
# tokenizer = text.Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(list(list_sentences_train))
#
# list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
# list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
#
# X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
# X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


def model():
    embed_size = 128
    input = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(input)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(input=input, output=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

    #
    # x = Embedding(max_features, embed_size)(inp)
    # x = Bidirectional(LSTM(50, return_sequences=True))(x)
    # x = GlobalMaxPool1D()(x)
    # x = Dropout(0.1)(x)
    # x = Dense(50, activation="relu")(x)
    # x = Dropout(0.1)(x)
    # x = Dense(6, activation="sigmoid")(x)
    # model = Model(inputs=inp, outputs=x)
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    # return model

model()