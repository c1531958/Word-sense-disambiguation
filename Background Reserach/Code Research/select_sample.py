import pandas as pd

RATIO_INDEX = 4


def training_sample(training_data):
    """
    In some cases, for example, for the word 'pitcher' there are 6403 training samples for the label 0 and only
    18 for the label 1. This can easily (especially classifier like knn) to lead to ignore one of the labels completely
    and still give high accuracy. To prevent this from happening and to see if it would improve the results,
    this function allows to toy around with label ratios.

    This function finds the minimum amount of labels (eg for 'pitcher it would be 18) and ensures that the new selected
    data sample has no more labels for the other classes than the multiplication of minimum classes and RATIO_INDEX.
    For example for pitcher it would be 4 * 18 = 72 labels for class 0 and 18 for class 1.
    """
    count_labels = training_data['label'].value_counts()
    min_count = count_labels.min()
    ratio = min_count * RATIO_INDEX
    new_train = pd.DataFrame()
    n_ = None
    for i in range(len(count_labels)):
        if count_labels[i] < ratio:
            n_ = count_labels[i]
        else:
            n_ = ratio

        sample = training_data[training_data['label'] == i].sample(n=n_, random_state=42)
        if new_train.empty:
            new_train = sample
        else:
            new_train = new_train.append(sample)

    return new_train
