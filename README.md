yahoo-news-titles
-----
- Yahoo!ニュース カテゴリー別記事タイトル一覧データ
- 機械学習によるY!ニュースタイトルのカテゴリ分類スクリプト

# ファイル
- 20171012 < Y!ニュースタイトルデータ
  - computer.txt
  - domestic.txt
  - economy.txt
  - entertainment.txt
  - local.txt
  - science.txt
  - sports.txt
  - world.txt
- fetch.py < Y!ニュースタイトル取得用スクリプト
- train.py < 学習用スクリプト
- predict.py < ニュースタイトル予測スクリプト

# 必要なもの
- Python3
- MeCab

# 使い方
依存パッケージインストール
```bash
$ pip install -r requirements.txt
```

Y!ニュースタイトルカテゴリ分類用学習モデルを作成する
```bash
$ python train.py
教師データを読み込みました: titles=2000 labels=2000

形態素解析によりトークン化しました: ドキュメント数=2000

辞書を作成しました: ユニークトークン数=541

辞書を保存しました: dictionary.dict

BOWコーパスを作成しました: [(0, 1), (1, 1), (2, 1), (3, 1)]

TF-IDFモデルを作成しました

TF-IDFモデルを保存しました: tfidf.model

TF-IDFコーパスを作成しました: [(0, 0.6000510631810851), (1, 0.4612509915120331), (2, 0.5841281844578962), (3, 0.29322433140232035)]

トレーニングデータ 1400件, テストデータ 600件

SVM学習モデルの作成を開始します ===>
Fitting 3 folds for each of 16 candidates, totalling 48 fits
[CV] C=0.1, gamma=0.1 ................................................
[CV] ....... C=0.1, gamma=0.1, score=0.5096359743040685, total=   0.8s
...
[CV] C=10, gamma=10 ..................................................
[CV] ......... C=10, gamma=10, score=0.5643776824034334, total=   0.9s
[Parallel(n_jobs=1)]: Done  48 out of  48 | elapsed:  1.0min finished
<=== SVM学習モデルの作成が終了しました

Cs: [0.1 0.1 0.1 0.1 1 1 1 1 5 5 5 5 10 10 10 10]
gammas: [0.1 1 5 10 0.1 1 5 10 0.1 1 5 10 0.1 1 5 10]
scores: [ 0.51        0.52714286  0.52714286  0.52714286  0.80642857  0.83        0.625
  0.54857143  0.82714286  0.82714286  0.65214286  0.55214286  0.825
  0.82357143  0.65214286  0.55214286]

best params: C=1, gamma:1
best score: 0.83
test score: 0.873333333333

SVM学習モデルを保存しました: svc.pkl
```

Y!ニュースタイトルカテゴリを予測してみる（モデル作成後）
```bash
$ python predict.py
辞書を読み込みました: dictionary.dict
TF-IDFモデルを読み込みました: tfidf.model
SVM学習モデルを読み込みました: svc.pkl

予測したいニュースタイトルを入力してください...

ドコモが2画面スマホ なぜ今写真
-----
入力: ドコモが2画面スマホ なぜ今写真
予測: computer
```

# Macでmatplotlibを使う
```bash
$ echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc
```
