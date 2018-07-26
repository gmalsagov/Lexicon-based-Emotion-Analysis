import discourse_mapper

import csv
import english_stoplist
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import re

emotions = ['anticipation', 'joy', 'negative', 'sadness', 'disgust', 'positive', 'anger', 'surprise', 'fear', 'trust']


def strip_punctuation(s):
    """Strip punctuation from a string"""
    return re.sub("[\.\t\,\:;\(\)\.]", "", s, 0, 0)


def process(text):
    """
    This function does several fuctions for text preparation:

    * Cleaning the string
    * Tokenization
    * Application of a stop-list
    * Stemming (snowball in English)

    Args:
    text : (string or clean Unicode) text to be processed

    Returns:
    tokens: a list of tokens where token is a stemmed word

    """

    tokens = word_tokenize(strip_punctuation(text.lower()))
    out_tokens = []

    stoplist = english_stoplist.stoplist
    stemmer = SnowballStemmer('english')

    for token in tokens:
        if (token not in stoplist) and (not token.startswith('http')) and (len(token) > 3):
            tt = stemmer.stem(token)
            # print(tt)
            out_tokens.append(tt)

    # print(out_tokens)
    return out_tokens


def clean_text(text):
    s=re.sub('<[^<]+?>', '', text).lower()
    try: 
        return str(s).decode('utf8',errors='ignore')
    except:
        return ""


dm = discourse_mapper.discourse_mapper()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def train():

    lexicon = csv.reader(open('data/NRC/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'),delimiter='\t')
    for row in lexicon:
        if len(row) < 3:
            continue
        if row[2] != '0':
            token = process(row[0])
            # print(token)
            if not token:
                continue
            # print row[1], token[0]
            dm.wb.add_word(row[1], token[0])
            # print len(dm.wb.word_graph)


def sorted_dic(d):
    ds = sorted(d.iteritems(), key=lambda (k,v): (-v,k))
    return ds


def predict(author, text):
    dm.add_text(author, text)
    return sorted_dic({emotion: dm._get_sentiment(author, emotion) for emotion in emotions})


train()

emotionss = []
for emotion in predict("Mark Lawrenson", u"That's a decent header but it's a very good save. Robin Olsen gets down low "
                                          u"to his left-hand side and he made sure he got everything behind it."):
    print emotion[0] + ': ' + str(emotion[1])


