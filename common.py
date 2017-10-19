import MeCab
from gensim import corpora, models, matutils
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

mecab = MeCab.Tagger()
mecab.parse('')

CATEGORIES = {
    0: 'computer',
    1: 'domestic'
}

DICT_PATH = 'dictionary.dict'
TFIDF_MODEL_PATH = 'tfidf.model'
SVC_MODEL_PATH = 'svc.pkl'

def tokenize(text):
    node = mecab.parseToNode(text)
    tokens = []
    while node:
        features = node.feature.split(',')
        if features[0] != 'BOS/EOS':
            tokens.append(node.surface)
        node = node.next
    return tokens

def load_file(path):
    with open(path, 'r') as f:
        lines = [line.rstrip('\r\n') for line in f]
    return lines

def load_documents():
    titles = []
    labels = []
    for idx, category in CATEGORIES.items():
        ttls = load_file('20171012/%s.txt' % (category))
        titles.extend(ttls)
        labels.extend([idx for _ in range(len(ttls))])
    return titles, labels
