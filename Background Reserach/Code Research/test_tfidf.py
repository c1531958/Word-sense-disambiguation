from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

punct =['.',',',';',':','!','\'', '?', '"', '(', ')', '[', ']', '<', '>', '\\', '/']
english_stop_words = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

words = ['hood', 'java', 'mole', 'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk', 'yard']

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


	# merge train date with labels data in one table
	train = pd.merge(train_text, train_label, left_index=True, right_index=True)
	test = pd.merge(test_text, test_label, left_index=True, right_index=True)


	# preprocess the sentences
	# shouldnt matter too much as tfidf should assign low scores to stop words
	for i in range(0, len(train.sentence)):
		sentence_tokens = get_list_tokens(train.sentence[i])
		rm_st = remove_stop_words(sentence_tokens)
		rm_punct = remove_punct(rm_st)
		# update cell
		train.at[i,'sentence']=' '.join(rm_punct)

	for i in range(0, len(test.sentence)):
		sentence_tokens = get_list_tokens(test.sentence[i])
		rm_st = remove_stop_words(sentence_tokens)
		rm_punct = remove_punct(rm_st)
		# update cell
		test.at[i,'sentence']=' '.join(rm_punct)



	list_rows = train['sentence'].tolist()
	vectorizer = TfidfVectorizer()
	# fit only
	X = vectorizer.fit(list_rows)

	# kkn(xtrain, ytarin)
	# knn = KNeighborsClassifier(n_neighbors=len(train.label.unique()))
	knn = KNeighborsClassifier(n_neighbors=3)

	knn.fit(X.transform(list_rows).todense(), train.label)
	prediction = []
	# print(knn)

	for sentence in test.sentence:
		p = knn.predict(X.transform([sentence]).todense())
		# convert from numpy nd array to int
		prediction.append(int(p[0]))

	accuracy = accuracy_score(test.label, prediction)
	precision, recall, fscore, support = precision_recall_fscore_support(test.label, prediction, average='macro', zero_division=0)
	stats = [accuracy, precision, recall, fscore]
	stats_all.append(stats)
	print(word)
	print(accuracy)

df = pd.DataFrame(stats_all, columns=['accuracy', 'precision', 'recall', 'fscore'], index=words)
print(df)
