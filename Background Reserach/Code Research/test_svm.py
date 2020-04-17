import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm

import time
import error_stats

punct =['.',',',';',':','!','\'', '?', '"', '(', ')', '[', ']', '<', '>', '\\', '/']
english_stop_words = stopwords.words('english')
words = ['hood', 'java', 'mole', 'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk', 'yard']
lemmatizer = nltk.stem.WordNetLemmatizer()

t0 = time.time()

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

def remove_punct(list_tokens):
	no_punct = [i for i in list_tokens if i not in punct]
	return no_punct


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
	for i in range(0,1):
		# print(train.sentence[i])
		sentence_tokens = get_list_tokens(train.sentence[i])
		rm_st = remove_stop_words(sentence_tokens)
		rm_punct = remove_punct(rm_st)
		train.at[i,'sentence'] = ' '.join(rm_punct)
		# print(train.sentence[i])
	for i in range(0,1):
		# print(test.sentence[i])
		sentence_tokens = get_list_tokens(test.sentence[i])
		rm_st = remove_stop_words(sentence_tokens)
		rm_punct = remove_punct(rm_st)
		test.at[i,'sentence'] = ' '.join(rm_punct)
		# print(test.sentence[i])

	list_rows = train['sentence'].tolist()
	# print(list_rows)
	vectorizer = TfidfVectorizer(max_features=1000)
	X = vectorizer.fit_transform(list_rows)
	X = X.toarray()
	# print(X)
	SVM_analysis = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	SVM_analysis.fit(X,train.label)

	prediction = []
	test_rows = test['sentence'].tolist()
	Y = vectorizer.transform(test_rows)
	Y = Y.toarray()
	prediction = SVM_analysis.predict(Y)

	# accuracy = accuracy_score(test.label, prediction)
	# precision, recall, fscore, support = precision_recall_fscore_support(test.label, prediction, average='macro')
	# stats = [accuracy, precision, recall, fscore]
	# stats_all.append(stats)
	# print(word)
	# print(accuracy)

	# get the confusion matrix
	ax = error_stats.format_conf_matrix(train, test, prediction, word, words)
	# get stats (accuracy, precision etc)
	stats = error_stats.get_stats(test.label, prediction)
	stats_all.append(stats)

t1 = time.time()
print('Time it took: {}'.format(t1 - t0))

df = pd.DataFrame(stats_all, columns=['accuracy', 'precision', 'recall', 'fscore'], index=words)
print(df)

