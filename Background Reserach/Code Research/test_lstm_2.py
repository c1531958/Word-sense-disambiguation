import nltk
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors

from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.utils import class_weight
from preprocess import preprocess
import error_stats
import numpy as np
import time
from sklearn.metrics import confusion_matrix

word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', \
        binary=True)

STOPWORDS = set(stopwords.words('english'))

punct = ['.', ',', ';', ':', '!', '\'', '?', '"', '(', ')', '[', ']', '<', '>', '\\', '/']
english_stop_words = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

words = ['hood', 'java', 'mole', 'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk', 'yard']

stats_all = []
conf_matrix_all = []

t0 = time.time()

def convertTow2v(list_tokens, selected_features, w2v_vector):
    embedding_matrix = np.zeros((len(selected_features), 300))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)

    # w2v = np.zeros((len(list_tokens), len(selected_features), 100))
    # set_s_f = set(selected_features)
    # for i in range(len(list_tokens)):
    #     # in case some training data tokens are not in features
    #     inters = set(list_tokens.iloc[i]).intersection(set_s_f)
    #     for word in inters:
    #         j = selected_features.index(word)
    #         w2v[i][j] = w2v_vector[j]

    # w2v = np.sum(w2v, axis=1)

    return embedding_matrix

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
    # vectorizer = TfidfVectorizer(max_features=1000)

    # X = vectorizer.fit(list_rows)
    # selected_features = vectorizer.get_feature_names()

    # model = Word2Vec(list_tokens, size=100, window=5, min_count=1)
    # model.train(list_tokens, total_examples=len(list_tokens), epochs=30)


    # try bow
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    EMBEDDING_DIM = 100
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(train['sentence'].values)
    word_index = tokenizer.word_index

    embedding_matrix = np.zeros((len(word_index)+1, 300))
    for key, i in word_index.items():
        if key in word2vec.wv.vocab:
            embedding_matrix[i] = word2vec.wv[key]

    # w2v_vector = model[selected_features]
    # w2v = convertTow2v(list_tokens, selected_features, w2v_vector, word_index)
    # w2v_test = convertTow2v(test_tokens, selected_features, w2v_vector)

    X_ = tokenizer.texts_to_sequences(train['sentence'].values)
    # print(X_.idf_)
    # b = np.rint(np.sort(X.transform(list_rows).toarray()*100))
    # print(b[0])
    
    # print(len(X[0]))
    X_ = pad_sequences(X_, maxlen=MAX_SEQUENCE_LENGTH)
    # print(X_[0])
    # print(type(X))
    # print(X.shape)


    X_test = tokenizer.texts_to_sequences(test['sentence'].values)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    # print(X_test[0])

    label_types = train.label.nunique()

    model = Sequential()
    model.add(Embedding(len(embedding_matrix), 300, weights=[embedding_matrix])) #, input_length=X.shape[1]
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(label_types, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 30
    batch_size = 64

    history = model.fit(X_, pd.get_dummies(train['label']).values, #pd.get_dummies(train['label']).values
                        class_weight=class_weights,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss',
                                                 patience=5,
                                                 min_delta=0.0001,
                                                 restore_best_weights=True)])

    # prediction = model.predict_classes(X.transform(test.sentence).todense())
    prediction = model.predict_classes(X_test)
    # accr = model.evaluate(X.transform(test.sentence).todense(), pd.get_dummies(test['label']).values)
    # print(accr[1])
    print(prediction)
    print(word)
    # get the confusion matrix
    ax = error_stats.format_conf_matrix(train, test, prediction, word, words)
    # get stats (accuracy, precision etc)
    stats = error_stats.get_stats(test.label, prediction)
    stats_all.append(stats)
    break

t1 = time.time()

print('Time it took: {}'.format(t1-t0))
df = pd.DataFrame(stats_all, columns=['accuracy', 'precision', 'recall', 'fscore'], index=words)
print(df)
plt.show()
