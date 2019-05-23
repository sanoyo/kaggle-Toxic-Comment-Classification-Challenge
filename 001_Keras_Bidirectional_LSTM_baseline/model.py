"""
全体参考
https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras
"""

"""
Embedding参考
http://yagami12.hatenablog.com/entry/2017/12/30/175113#ID_10-4
"""

import numpy as np
import pandas as pd

from subprocess import check_output
print(check_output(["ls", "../csv"]).decode("utf8"))


from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pdb


max_features = 20000
# 長い文章から作ったindexに制限をつける
maxlen = 100

train = pd.read_csv("../csv/train.csv")
test = pd.read_csv("../csv/test.csv")
train = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

# テキストをベクトルに変換
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))

# indexに変換
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# 短いindexを同じ長さにするためにpaddingを行う
# maxlenで設定されている100になるよう値を0で埋める
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

"""
サンプルデータ
 X_t[8]
array([    0,     0,     0,     0,     0,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,   263,    22,
           1,   286,    24,  1476,     2,     6,   489,    73,    14,
        9215,     2,   330,   215,    10,     1,    23,  1473,    51,
          47,  2898,    15,    35,    12,   199,    73,  1051,  1060,
           9,    11,    16,    57,  1390,    37,    48,    40,    81,
          11,    12,   480,    17,     5,   273,     7,    18,    55,
           2,     1,  4850,  8153,    27,    26,    78,   669,     5,
        1368,    11,   687,     2,    53,     8,     5,   702,  7774,
          23,     9,  1094,     6,    46,   468,    41,   251,    16,
         159,   324,    21,  3566,    10, 13824,  3439,  4584,  3890,
        4851], dtype=int32)

"""


def get_model():
    embed_size = 128
    # 最初に定義したmaxlen(最大のindexの長さ)がinputの大きさになる
    inp = Input(shape=(maxlen, ))
    #
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# 学習
model = get_model()
batch_size = 32
epochs = 50

file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early]
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

# 評価
model.load_weights(file_path)
y_test = model.predict(X_te)

sample_submission = pd.read_csv("../csv/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("baseline.csv", index=False)