import os
import random
import MeCab
from gensim import corpora, models, matutils
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
mecab.parse('')

categories = {
    0: 'computer',
    1: 'domestic'
}

def tokenize(text):
    node = mecab.parseToNode(text)
    tokens = []
    while node:
        features = node.feature.split(',')
        if features[0] != 'BOS/EOS':
            tokens.append(node.surface)
        node = node.next
    return tokens

dictionary = corpora.Dictionary.load('dictionary.dict')
print('dictionary: uniq tokens=%s' % (len(dictionary)))

titles = ['PC不要のVR端末を発表 米FB']
documents = [tokenize(title) for title in titles]

bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
print('bow_corpus:', bow_corpus[0])

tfidf = models.TfidfModel.load('tfidf.model')
tfidf_corpus = tfidf[bow_corpus]
print('tfidf_corpus:', tfidf_corpus[0])

x_test = [matutils.corpus2dense([corpus], num_terms=len(dictionary)).T[0] for corpus in tfidf_corpus]

clf = joblib.load('svc.pkl')
for i in range(len(titles)):
    title = titles[i]
    result = clf.predict([x_test[i]])[0]
    print('')
    print('title:', title)
    print('predict:', categories[result])
