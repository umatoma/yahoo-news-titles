import sys
from gensim import corpora, models, matutils
from sklearn.externals import joblib
from common import CATEGORIES, DICT_PATH, TFIDF_MODEL_PATH, SVC_MODEL_PATH, tokenize

dictionary = corpora.Dictionary.load(DICT_PATH)
print('辞書を読み込みました:', DICT_PATH)

tfidf = models.TfidfModel.load(TFIDF_MODEL_PATH)
print('TF-IDFモデルを読み込みました:', TFIDF_MODEL_PATH)

clf = joblib.load(SVC_MODEL_PATH)
print('SVM学習モデルを読み込みました:', SVC_MODEL_PATH)

print('')
print('予測したいニュースタイトルを入力してください...')
print('')

try:
    for line in sys.stdin:
        title = line.rstrip('\r\n')
        documents = [tokenize(title)]
        bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
        tfidf_corpus = tfidf[bow_corpus]
        X = [matutils.corpus2dense([corpus], num_terms=len(dictionary)).T[0] for corpus in tfidf_corpus]

        result = clf.predict(X)[0]
        print('-----')
        print('入力:', title)
        print('予測:', CATEGORIES[result])
        print('')
except KeyboardInterrupt:
    print('===== 終了 =====')
