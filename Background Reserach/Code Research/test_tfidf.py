import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import class_weight
from sklearn.metrics import plot_confusion_matrix

import error_stats
from preprocess import preprocess
from select_sample import training_sample

from sklearn.utils import class_weight


stats_all = []

words = ['hood', 'java', 'mole', 'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk', 'yard']
words=['pitcher']
t0 = time.time()

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

    # merge train date with labels data in one table
    train = pd.merge(train_text, train_label, left_index=True, right_index=True)
    test = pd.merge(test_text, test_label, left_index=True, right_index=True)

    # get class weights
    # class_weights = class_weight.compute_class_weight('balanced',
    #                                                   np.unique(train.label),
    #                                                   train.label)
    # print(class_weights)

    # if there are enough training samples, even the label ratios out
    if train.shape[0] > 1000:
        train = training_sample(train)

    # preprocess the sentences
    # does not matter too much as tfidf assigns low scores to stop words
    train['sentence'] = preprocess(train)
    test['sentence'] = preprocess(test)

    list_rows = train['sentence'].tolist()
    vectorizer = TfidfVectorizer()
    # if there are more than 1000 training samples, limit the max_features to 1000
    if train.shape[0] > 1000:
        vectorizer.max_features = 1000

    # fit only
    X = vectorizer.fit(list_rows)

    # get class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train.label),
                                                      train.label)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X.transform(list_rows).todense(), train.label)

    prediction = knn.predict(X.transform(test.sentence).todense())

    # get the confusion matrix
    ax = error_stats.format_conf_matrix(train, test, prediction, word, words)
    # get stats (accuracy, precision etc)
    stats = error_stats.get_stats(test.label, prediction)
    stats_all.append(stats)

t1 = time.time()

print('Time it took: {}'.format(t1 - t0))
df = pd.DataFrame(stats_all, columns=['accuracy', 'precision', 'recall', 'fscore'], index=words)
print(df)
plt.show()
