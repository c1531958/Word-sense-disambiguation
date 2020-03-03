import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import io
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import nltk

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

english_stop_words = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


train_text = pd.read_csv('../CoarseWSD_P2/hood/train.data.txt',
                    sep='\t',
                    names=['index', 'sentence'])
train_label = pd.read_csv('../CoarseWSD_P2/hood/train.gold.txt',
                    sep='\t',
                    names=['label'])

train = pd.merge(train_text, train_label, left_index=True, right_index=True)
train.head()

# column used for prediction process
pediction_column = train['label'].values
# preprocessing scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pediction_column.reshape(-1,1))

def get_list_tokens(string):
    sentence_split = nltk.tokenize.sent_tokenize(string)
    list_tokens = []
    for sentence in sentence_split:
        list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            list_tokens.append(lemmatizer.lemmatize(token).lower())

    return list_tokens

def remove_stop_words(list_tokens):
    clean_list_tokens = []
    for token in list_tokens:
        if token not in english_stop_words:
            clean_list_tokens.append(token)

    return clean_list_tokens

punct =['.',',',';',':','!','\'', '?', '"', '(', ')', '[', ']', '<', '>', '\\', '/']
def remove_punct(list_tokens):
    no_punct = [i for i in list_tokens if i not in punct]
    return no_punct

def getVocabulary(training_set):
    dict_frequency = {}
    vocabulary = []
    for review in training_set:
        sentence_tokens = get_list_tokens(review)
        rm_st = remove_stop_words(sentence_tokens)
        rm_punct = remove_punct(rm_st)
        vocabulary += rm_punct
    vocabulary = list(set(vocabulary))
    return vocabulary

training_set = train.sentence
vocabulary = getVocabulary(training_set)
# print(training_set)

# binary encode an input pattern, return a list of binary vectors
def encode(pattern, n_unique):
	encoded = list()
	for word in pattern:
		value = vocabulary.index(word)
		row = [0.0 for x in range(n_unique)]
		row[value] = 1.0
		encoded.append(row)
	return encoded

# create input/output pairs of encoded vectors, returns X, y
def to_xy_pairs(encoded):
	X,y = list(),list()
	for i in range(1, len(encoded)):
		X.append(encoded[i-1])
		y.append(encoded[i])
	return X, y

# convert sequence to x/y pairs ready for use with an LSTM
encoder = OneHotEncoder()
def to_lstm_dataset(sequence, n_unique):
	# one hot encode
	encoded = encode(sequence, n_unique)
	# convert to in/out patterns
	X,y = to_xy_pairs(encoded)
	# convert to LSTM friendly format
	dfX, dfy = pd.DataFrame(X), pd.DataFrame(y)
	lstmX = dfX.values
	lstmX = lstmX.reshape(lstmX.shape[0], 1, lstmX.shape[1])
	lstmY = dfy.values
	return lstmX, lstmY

n_unique = len(vocabulary)

# define LSTM configuration
n_neurons = 20
n_batch = 1
n_epoch = 100
n_features = n_unique
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, 1, n_features), stateful=True))
model.add(Dense(n_unique, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

for sentence in train.sentence[:20]:
    # define sequences
    sentence_tokens = get_list_tokens(train.sentence[0])
    rm_st = remove_stop_words(sentence_tokens)
    seq1 = remove_punct(rm_st)
    # convert sequences into required data format
    seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
    for i in range(n_epoch):
        model.fit(seq1X, seq1Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
        model.reset_states()
        # model.fit(seq2X, seq2Y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        # model.reset_states()

# test LSTM on sequence 1
print('Sequence 1')
sentence_tokens = get_list_tokens(train.sentence[3])
rm_st = remove_stop_words(sentence_tokens)
seq1 = remove_punct(rm_st)
seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)

result = model.predict_classes(seq1X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X={} y={}, yhat={}'.format(seq1[i], seq1[i+1], vocabulary[result[i]]))
print(result)