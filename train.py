from gensim import corpora, models, matutils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib
from sklearn.svm import SVC
from common import CATEGORIES, DICT_PATH, TFIDF_MODEL_PATH, SVC_MODEL_PATH, tokenize, load_documents

titles, labels = load_documents()
print('教師データを読み込みました: titles=%s labels=%s' % (len(titles), len(labels)))
print('')

documents = [tokenize(title) for title in titles]
print('形態素解析によりトークン化しました: ドキュメント数=%s' % (len(documents)))
print('')

dictionary = corpora.Dictionary(documents)
dictionary.filter_extremes(no_above=0.8)
print('辞書を作成しました: ユニークトークン数=%s' % (len(dictionary)))
print('')

dictionary.save(DICT_PATH)
print('辞書を保存しました:', DICT_PATH)
print('')

bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
print('BOWコーパスを作成しました:', bow_corpus[0])
print('')

tfidf = models.TfidfModel(bow_corpus)
print('TF-IDFモデルを作成しました')
print('')

tfidf.save(TFIDF_MODEL_PATH)
print('TF-IDFモデルを保存しました:', TFIDF_MODEL_PATH)
print('')

tfidf_corpus = tfidf[bow_corpus]
print('TF-IDFコーパスを作成しました:', tfidf_corpus[0])
print('')

X = [matutils.corpus2dense([corpus], num_terms=len(dictionary)).T[0] for corpus in tfidf_corpus]
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print('トレーニングデータ %s件, テストデータ %s件' % (len(X_train), len(X_test)))
print('')

print('SVM学習モデルの作成を開始します ===>')
svc = SVC(kernel='rbf')
C_range = [0.1, 1, 5, 10]
gamma_range = [0.1, 1, 5, 10]
clf = GridSearchCV(svc, dict(C=C_range, gamma=gamma_range), verbose=3)
clf.fit(X_train, y_train)
print('<=== SVM学習モデルの作成が終了しました')
print('')

print('Cs:', clf.cv_results_['param_C'])
print('gammas:', clf.cv_results_['param_gamma'])
print('scores:', clf.cv_results_['mean_test_score'])
print('')

print('best params: C=%s, gamma:%s' % (clf.best_params_['C'], clf.best_params_['gamma']))
print('best score:', clf.best_score_)
print('test score:', clf.score(X_test, y_test))
print('')

joblib.dump(clf, SVC_MODEL_PATH)
print('SVM学習モデルを保存しました:', SVC_MODEL_PATH)
