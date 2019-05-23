"""
全体参考
https://www.kaggle.com/eashish/bidirectional-gru-with-convolution
"""

import numpy as np
import pandas as pd
import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pdb

maxlen = 128
max_features = 20000
embed_size = 200

batch_size = 128
epochs = 4

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

def cnn_lstm():
    input = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(input)
    x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = SpatialDropout1D(0.2)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    preds = Dense(6, activation="sigmoid")(x)
    model = Model(input, preds)

    return model


df_train = pd.read_csv("../csv/train.csv")
train = df_train["comment_text"]

df_test = pd.read_csv("../csv/test.csv")
test = df_test["comment_text"]

train = train.fillna("CVxTz").values
test = test.fillna("aaa").values

lists = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df_train[lists].values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train)


tokenized_train = tokenizer.texts_to_sequences(train)
tokenized_test = tokenizer.texts_to_sequences(test)

X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

model = cnn_lstm()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


X_train, X_val, y_train, y_val = train_test_split(X_train, y, train_size=0.9, random_state=23)

file_path="model/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
callbacks_list = [ra_val,checkpoint, early]

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks_list, verbose=1)
