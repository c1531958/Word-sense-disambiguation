from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# 0 = comics/not a part of wordnet
# 1 = vehicle/Synset('hood.n.09') protective covering consisting of a metal part that covers the engine
# 2 = headgear/Synset('hood.n.08') a headdress that protects the head and face
# for ss in wn.synsets('hood'):
#     print(ss, ss.definition())

# returns mapping from the given definitions to definitions retrurned by wordnet. Some meanings are missing in wordnet.
def mapToLeskEquivalent(word):
    if word == 'hood':
        return {wn.synset('hood.n.09') : 1, wn.synset('hood.n.08') : 2}
    elif word == 'java':
        return {wn.synset('java.n.01') : 0, wn.synset('java.n.03') : 1}
    elif word == 'mole':
        return {wn.synset('mole.n.06') : 0, wn.synset('counterspy.n.01') : 1, wn.synset('gram_molecule.n.01') : 2, wn.synset('mole.n.03') : 3,
                wn.synset('breakwater.n.01') : 4}
    elif word == 'pitcher':
        return {wn.synset('pitcher.n.01') : 0, wn.synset('pitcher.n.02') : 1}
    elif word == 'pound':
        return {wn.synset('pound.n.03') : 0, wn.synset('british_pound.n.01') : 1}
    elif word == 'seal':
      return {wn.synset('seal.n.09') : 0, wn.synset('sealing_wax.n.01') : 2, wn.synset('seal.n.08') : 3}
    elif word == 'spring':
        return {wn.synset('spring.n.03') : 0, wn.synset('spring.n.01') : 1, wn.synset('spring.n.02') : 2}
    elif word == 'square':
        return {wn.synset('square.n.01') : 0, wn.synset('public_square.n.01') : 2, wn.synset('square.n.02') : 3}
    elif word == 'trunk':
        return {wn.synset('trunk.n.01') : 0, wn.synset('trunk.n.02') : 1, wn.synset('torso.n.01') : 2}
    elif word == 'yard':
        return {wn.synset('yard.n.01') : 0, wn.synset('yard.n.08') : 1}

path = '../CoarseWSD_P2/hood/classes_map.txt'
words = ['hood', 'java', 'mole', 'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk', 'yard']
stats_all = []

for word in words:
    test_set_path = '../CoarseWSD_P2/{}/test.data.txt'.format(word)
    test_set_gold_path = '../CoarseWSD_P2/{}/test.gold.txt'.format(word)

    # reads txt file and converts it into a dictionary
    classes = eval(open(path, encoding="utf8").read())
    test_set = open(test_set_path, encoding="utf8").read().splitlines()
    test_set_labels_str = open(test_set_gold_path, encoding="utf8").read().splitlines()
    test_set_labels = [int(i) for i in test_set_labels_str]

    word_map = mapToLeskEquivalent(word)

    prediction_lesk = []
    for i in range(len(test_set)):
        context = test_set[i].split()
        label = test_set_labels[i]
        # n stands for noun
        prediction = lesk(context, word, 'n')
        prediction_lesk.append(word_map.get(prediction, -1))


    accuracy = accuracy_score(test_set_labels, prediction_lesk)
    precision, recall, fscore, support = precision_recall_fscore_support(test_set_labels, prediction_lesk, average='macro', zero_division=0)
    stats = [accuracy, precision, recall, fscore]
    stats_all.append(stats)
    
df = pd.DataFrame(stats_all, columns=['accuracy', 'precision', 'recall', 'fscore'], index=words)
print(df)