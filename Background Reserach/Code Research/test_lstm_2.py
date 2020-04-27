import time

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.utils import class_weight

import error_stats
from preprocess import preprocess

word2vec = api.load('word2vec-google-news-300')

words = ['hood', 'java', 'mole', 'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk', 'yard']

stats_all = []

t0 = time.time()

for word in words:
    train_text = pd.read_csv('../CoarseWSD_P2/{}/train.data.txt'.format(word),
                             sep='\t',
                             names=['index', 'sentence'])
    train_label = pd.read_csv('../CoarseWSD_P2/{}/train.gold.txt'.format(word),
                              sep='\t',
                              names=['label'])

    test_text = pd.read_csv('../CoarseWSD_P2/{}/test.data.txt'.format(word), sep='\t', names=['index', 'sentence'])
    test_label = pd.read_csv('../CoarseWSD_P2/{}/test.gold.txt'.format(word), sep='\t', names=['label'])

    # merge train date with labels data in one table
    train = pd.merge(train_text, train_label, left_index=True, right_index=True)
    test = pd.merge(test_text, test_label, left_index=True, right_index=True)

    # get class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train.label),
                                                      train.label)

    # preprocess the sentences
    train['sentence'] = preprocess(train)
    test['sentence'] = preprocess(test)

    list_tokens = train['sentence'].apply(lambda x: x.split(' '))
    test_tokens = test['sentence'].apply(lambda x: x.split(' '))
# 
    list_rows = train['sentence'].tolist()

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Highest sentence length within training data set  - 100
    MAX_SEQUENCE_LENGTH = train.sentence.map(len).max() - 100
    EMBEDDING_DIM = 100
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(train['sentence'].values)
    word_index = tokenizer.word_index

    embedding_matrix = np.zeros((len(word_index)+1, 300))
    for key, i in word_index.items():
        if key in word2vec.wv.vocab:
            embedding_matrix[i] = word2vec.wv[key]

    X_ = tokenizer.texts_to_sequences(train['sentence'].values)
    X_ = pad_sequences(X_, maxlen=MAX_SEQUENCE_LENGTH)

    X_test = tokenizer.texts_to_sequences(test['sentence'].values)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    label_types = train.label.nunique()

    model = Sequential()
    model.add(Embedding(len(embedding_matrix), 300,  weights=[embedding_matrix]))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(label_types, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 30
    batch_size = 30

    history = model.fit(X_, pd.get_dummies(train['label']).values,
                        class_weight=class_weights,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss',
                                                 patience=5,
                                                 min_delta=0.0001)])

    prediction = model.predict_classes(X_test)
    print(prediction)
    print(word)
    # get the confusion matrix
    ax = error_stats.format_conf_matrix(train, test, prediction, word, words)
    # get stats (accuracy, precision etc)
    stats = error_stats.get_stats(test.label, prediction)
    stats_all.append(stats)

t1 = time.time()

print('Time it took: {}'.format(t1-t0))
df = pd.DataFrame(stats_all, columns=['accuracy', 'precision', 'recall', 'fscore', 'rmse'], index=words)
print(df)
plt.show()
