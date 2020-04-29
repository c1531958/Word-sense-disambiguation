import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

import time
import error_stats
from preprocess import preprocess
from select_sample import training_sample

import matplotlib.pyplot as plt

words = ['hood', 'java', 'mole', 'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk', 'yard']

t0 = time.time()

stats_all = []
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
    train = pd.merge(train_text, train_label, left_index=True, right_index=True)
    test = pd.merge(test_text, test_label, left_index=True, right_index=True)
    train['sentence'] = preprocess(train)
    test['sentence'] = preprocess(test)

    # if there are enough training samples, even the label ratios out
    if train.shape[0] > 1000:
        train = training_sample(train)

    list_rows = train['sentence'].tolist()
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(list_rows)
    X = X.toarray()
    SVM_analysis = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight='balanced')
    SVM_analysis.fit(X,train.label)

    test_rows = test['sentence'].tolist()
    Y = vectorizer.transform(test_rows)
    Y = Y.toarray()
    prediction = SVM_analysis.predict(Y)

    # get the confusion matrix
    ax = error_stats.format_conf_matrix(train, test, prediction, word, words)
    # get stats (accuracy, precision etc)
    stats = error_stats.get_stats(test.label, prediction)
    stats_all.append(stats)

t1 = time.time()
print('Time it took: {}'.format(t1 - t0))

df = pd.DataFrame(stats_all, columns=['accuracy', 'precision', 'recall', 'fscore', 'rmse'], index=words)
print(df)
plt.show()
