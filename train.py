import os
import random
import MeCab
from gensim import corpora, models, matutils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
mecab.parse('')

categories = {
    0: 'computer',
    1: 'domestic'
}

def load_file(path):
    with open(path, 'r') as f:
        lines = [line.rstrip('\r\n') for line in f]
    return lines

def load_data():
    titles = []
    labels = []
    for idx, category in categories.items():
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
dictionary.filter_extremes(no_above=0.8)
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
    print('classifier: load from pkl file')

    clf = joblib.load(pkl_path)
    for i in random.sample(range(len(titles)), 10):
        title = titles[i]
        label = labels[i]
        result = clf.predict([X[i]])[0]
        print('')
        print('title:', title)
        print('label:', categories[label])
        print('predict:', categories[result])
        print('true/false:', True if label == result else False)
else:
    print('classifier: create new model')

    svc = SVC(kernel='rbf')
    C_range = [0.1, 1, 5, 10]
    gamma_range = [0.1, 1, 5, 10]
    clf = GridSearchCV(svc, dict(C=C_range, gamma=gamma_range), verbose=3)
    clf.fit(X_train, y_train)

    print('Cs:', clf.cv_results_['param_C'])
    print('gammas:', clf.cv_results_['param_gamma'])
    print('scores:', clf.cv_results_['mean_test_score'])

    print('best params: C=%s, gamma:%s' % (clf.best_params_['C'], clf.best_params_['gamma']))
    print('best score:', clf.best_score_)
    print('test score:', clf.score(X_test, y_test))

    joblib.dump(clf, pkl_path)

    # HeatMap
    # scores = clf.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
    # plt.figure(figsize=(8, 6))
    # plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    # plt.xlabel('gamma')
    # plt.ylabel('C')
    # plt.colorbar()
    # plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    # plt.yticks(np.arange(len(C_range)), C_range)
    # plt.title('Validation accuracy')
    # plt.show()
