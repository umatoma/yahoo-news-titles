import os
import MeCab
from gensim import corpora, models, matutils
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.svm import SVC

mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
mecab.parse('')

categories = [
    (0, 'computer'),
    (1, 'domestic')
]

def load_file(path):
    with open(path, 'r') as f:
        lines = [line.rstrip('\r\n') for line in f]
    return lines

def load_data():
    titles = []
    labels = []
    for idx, category in categories:
        ttls = load_file('20171012/%s.txt' % (category))
        titles.extend(ttls)
        labels.extend([idx for _ in range(len(ttls))])
    return titles, labels

def tokenize(text):
    node = mecab.parseToNode(text)
    tokens = []
    while node:
        features = node.feature.split(',')
        if features[0] != 'BOS/EOS':
            tokens.append(node.surface)
        node = node.next
    return tokens

titles, labels = load_data()
print('data: titles=%s labels=%s' % (len(titles), len(labels)))

documents = [tokenize(title) for title in titles]
print('documents: size=%s' % (len(documents)))

dictionary = corpora.Dictionary(documents)
print('dictionary: uniq tokens=%s' % (len(dictionary)))

bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
print('bow_corpus:', bow_corpus[0])

tfidf = models.TfidfModel(bow_corpus)
tfidf_corpus = tfidf[bow_corpus]
print('tfidf_corpus:', tfidf_corpus[0])

X = [matutils.corpus2dense([corpus], num_terms=len(dictionary)).T[0] for corpus in tfidf_corpus]
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

pkl_path = 'svc.pkl'
if os.path.exists(pkl_path):
    clf = joblib.load(pkl_path)
    print('classifier: load from pkl file')
else:
    clf = SVC()
    print('classifier: create new model')
    clf.fit(X_train, y_train)
    joblib.dump(clf, pkl_path)
print('score:', clf.score(X_test, y_test))
