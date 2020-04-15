from gensim.models import Word2Vec, FastText
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from itertools import islice
import gensim.downloader as api

import error_stats
from preprocess import preprocess
from select_sample import training_sample

stats_all = []

words = ['hood', 'java', 'mole', 'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk', 'yard']
model2 = api.load('word2vec-google-news-300')

t0 = time.time()


def convertTow2v(list_tokens, selected_features, w2v_vector):
    w2v = np.zeros((len(list_tokens), len(selected_features), 300))
    set_s_f = set(selected_features)
    for i in range(len(list_tokens)):
        # in case some training data tokens are not in features
        inters = set(list_tokens.iloc[i]).intersection(set_s_f)
        for word in inters:
            j = selected_features.index(word)
            w2v[i][j] = w2v_vector[j]

    w2v = np.sum(w2v, axis=1)

    return w2v


for word in words:
    train_text = pd.read_csv('../CoarseWSD_P2/{}/train.data.txt'.format(word),
                             sep='\t',
                             names=['index', 'sentence'])
    train_label = pd.read_csv('../CoarseWSD_P2/{}/train.gold.txt'.format(word),
                              sep='\t',
                              names=['label'])

    test_text = pd.read_csv('../CoarseWSD_P2/{}/test.data.txt'.format(word),
                            sep='\t',
                            names=['index', 'sentence'])
    test_label = pd.read_csv('../CoarseWSD_P2/{}/test.gold.txt'.format(word),
                             sep='\t',
                             names=['label'])

    # merge train data with labels data in one table
    train = pd.merge(train_text, train_label, left_index=True, right_index=True)
    test = pd.merge(test_text, test_label, left_index=True, right_index=True)
    # if there are enough training samples, even the label ratios out
    if train.shape[0] > 1000:
        train = training_sample(train)

    # preprocess the sentences
    train['sentence'] = preprocess(train)
    test['sentence'] = preprocess(test)

    # if there are enough training samples, even the label ratios out
    if train.shape[0] > 1000:
        train = training_sample(train)

    list_tokens = train['sentence'].apply(lambda x: x.split(' '))
    test_tokens = test['sentence'].apply(lambda x: x.split(' '))

    # if there are more than 1000 training samples, limit the max_features to 1000 as otherwise it will exceed memory
    # try tfidf
    vectorizer = TfidfVectorizer()
    if train.shape[0] > 1000:
        vectorizer.max_features = 1000
    vectorizer.fit(train['sentence'])
    selected_features = vectorizer.get_feature_names()

    # try bow
    # tokenizer = Tokenizer(num_words=1000, lower=True)
    # tokenizer.fit_on_texts(train['sentence'].values)
    # selected_features = list(tokenizer.word_index.keys())[:1000]

    # w2v
    model = Word2Vec(list_tokens, size=300, window=5, min_count=1)
    # fast text
    # model = FastText(size=100, window=3, min_count=1)
    # model.build_vocab(sentences=list_tokens)
    model.train(list_tokens, total_examples=len(list_tokens), epochs=30)

    not_in_model = model2.doesnt_match(selected_features)
    w2v_vector = [model2[feature] if feature in not_in_model else model[feature] for feature in selected_features]

    # model = Word2Vec.load('word2vec-google-news-300')
    # not_in_model = model.doesnt_match(selected_features)
    # selected_features = selected_features - not_in_model
    # w2v_vector = model[selected_features]

    w2v = convertTow2v(list_tokens, selected_features, w2v_vector)
    w2v_test = convertTow2v(test_tokens, selected_features, w2v_vector)

    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(w2v, train.label)

    # for sentence in w2v_test:
    prediction = knn.predict(w2v_test)

    # get the confusion matrix
    ax = error_stats.format_conf_matrix(train, test, prediction, word, words)
    # get stats (accuracy, precision etc)
    stats = error_stats.get_stats(test.label, prediction)
    stats_all.append(stats)

t1 = time.time()

print('Time it took: {}'.format(t1-t0))
df = pd.DataFrame(stats_all, columns=['accuracy', 'precision', 'recall', 'fscore'], index=words)
print(df)
plt.show()
