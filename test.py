from os import listdir

import nltk, gensim
from linguist import process
from collections import Counter
from gensim import corpora, models, similarities

# specify directory to load
directory = 'data/txt_sentoken/neg'

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# load all docs in a directory
movie_reviews = []
# walk through all files in the folder
for filename in listdir(directory):
    # skip files that do not have the right extension
    if not filename.endswith(".txt"):
        continue
    # create the full path of the file to open
    path = directory + '/' + filename
    # load document
    doc = load_doc(path)
    movie_reviews.append(doc)


# for review in movie_reviews:
sent_text = [nltk.sent_tokenize(review) for review in movie_reviews]  # this gives us a list of sentences
# print(len(sent_text))
tokens = []
for review in sent_text:
    for sentence in review:
        tokens.append(process(sentence))
        # print(process(sentence))

frequencies = Counter()
for t in tokens:
    frequencies.update(t)

# print(frequencies.most_common(20))

# Remove words that occur only once
tokens = [[word for word in token if frequencies[word] > 1] for token in tokens]

dictionary = corpora.Dictionary(tokens)
dictionary.save('data/movie_reviews.dict')

print(dictionary.token2id)

corpus = [dictionary.doc2bow(token) for token in tokens]
corpora.MmCorpus.serialize('data/movie_review_corpus.mm', corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf(corpus)

# Latent Semantic Analysis
lsi = gensim.models.lsimodel.LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=400)
index = similarities.MatrixSimilarity(lsi[corpus])
index.save('data/movie_review.index')


# Function returning num most similar sentences in text
def text_lsi(new_text, num=10):
    new_vec = dictionary.doc2bow(process(new_text))
    vec_lsi = lsi[new_vec]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return[(s,movie_reviews[s[0]]) for s in sims[:num]]



#
# # load the document
# filename = 'data/txt_sentoken/neg/cv000_29416.txt'
#
# movie_reviews = load_doc(filename)
# sent_text = nltk.sent_tokenize(movie_reviews)  # this gives us a list of sentences
#
# tokens = [process(sentence) for sentence in sent_text]
#
# frequencies = Counter()
# for t in tokens: frequencies.update(t)
#
# print(frequencies)
# # split into tokens by white space
# tokens = text.split()
# # remove punctuation from each token
# table = str.maketrans('', '', string.punctuation)
# tokens = [w.translate(table) for w in tokens]
# # remove remaining tokens that are not alphabetic
# tokens = [word for word in tokens if word.isalpha()]
# # filter out stop words
# stop_words = set(stopwords.words('english'))
# tokens = [w for w in tokens if not w in stop_words]
# # filter out short tokens
# tokens = [word for word in tokens if len(word) > 1]
# print(tokens)