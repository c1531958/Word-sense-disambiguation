import nltk
from nltk.corpus import stopwords
import re

english_stop_words = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


def get_list_tokens(string):
    sentence_split = nltk.tokenize.sent_tokenize(string)
    for sentence in sentence_split:
        list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
        list_tokens = [lemmatizer.lemmatize(token).lower() for token in list_tokens_sentence]
    return list_tokens


def remove_stop_words(list_tokens):
    clean_list_tokens = [token for token in list_tokens if token not in english_stop_words]
    return clean_list_tokens


def remove_punct(list_tokens):
    no_punct = []
    for i in list_tokens:
        i = re.sub(r'[^\w\s]', '', i)
        if i and len(i) > 1:
            no_punct.append(i)

    return no_punct

def preprocess(df):
    df['sentence'] = df.apply(lambda x: ' '.join(
        remove_punct(
            remove_stop_words(
                get_list_tokens(x['sentence'])))), axis=1)

    return df['sentence']
