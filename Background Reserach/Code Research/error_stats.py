import itertools

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

TEXT_FONT_SIZE = 'medium'
fig = plt.figure(figsize=(30, 30))


def format_conf_matrix(train, test, prediction, word, words):
    """
    Some code taken from scikitplot library
    This function plots the confusion matrices
    """
    unique_labels = sorted(train.label.unique())
    ax = fig.add_subplot(2, 5, words.index(word) + 1)
    ax.set_title(word, fontsize='large')
    cm = confusion_matrix(test['label'].values, prediction, unique_labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.around(cm, decimals=2)
    cm[np.isnan(cm)] = 0.0

    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    ax.set_xticks(unique_labels)
    ax.set_xticklabels(unique_labels, fontsize=TEXT_FONT_SIZE)
    ax.set_yticks(unique_labels)
    ax.set_yticklabels(unique_labels, fontsize=TEXT_FONT_SIZE)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=TEXT_FONT_SIZE,
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label', fontsize=TEXT_FONT_SIZE)
    ax.set_xlabel('Predicted label', fontsize=TEXT_FONT_SIZE)
    ax.grid(False)


def get_stats(test_label, pred_label):
    accuracy = accuracy_score(test_label, pred_label)
    precision, recall, fscore, support = precision_recall_fscore_support(test_label, pred_label, average='macro',
                                                                         zero_division=0)
    rms = sqrt(mean_squared_error(test_label, pred_label))
    stats = [accuracy, precision, recall, fscore, rms]
    print(accuracy)

    return stats
