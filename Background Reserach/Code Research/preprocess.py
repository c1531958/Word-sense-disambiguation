import nltk
from nltk.corpus import stopwords
import re

punct = ['.', ',', ';', ':', '!', '\'', '?', '"', '(', ')', '[', ']', '<', '>', '\\', '/']
english_stop_words = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


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
    no_punct = []
    for i in list_tokens:
        i = re.sub(r'[^\w\s]', '', i)
        if i:
            no_punct.append(i)
    return no_punct


def preprocess(df):
    df['sentence'] = df.apply(lambda x: ' '.join(
        remove_punct(
            remove_stop_words(
                get_list_tokens(x['sentence'])))), axis=1)

    return df['sentence']
