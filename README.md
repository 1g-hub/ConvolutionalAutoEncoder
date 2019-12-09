Convolutional Auto Encoder サンプルレポジトリ

全体的に中身を読みつつ
適宜データとラベルの実装をしてください．

# Requirements
requirements.txtに記載.インストールするために以下を実行(色々といらないパッケージとかも入る気がする)．

```
$ pip install -r requirements.txt
```
# レポジトリをクローン
```
$ git clone https://github.com/1g-hub/ConvolutionalAutoEncoder
$ cd ConvolutionalAutoEncoder
```

## 各ファイルの役割
- autoencoder.py : 学習用のメインスクリプト
- classify.py : 分散表現からpredictを行う
- comic2vec.py : モデルを再構築して分散表現化する
- loader.py : 画像を読み込む．データローダー．
- model.py : さまざまなモデル定義をまとめている
- t_sne.py : 分散表現を2次元に圧縮したのち，可視化する

## 各フォルダの役割
学習の結果できるフォルダもある．
- decoded_imgs : エンコードされた画像とでコードされた画像を表示
- distributed:学習した重みをnumpy形式で保存
- figures:lossとt-SNEの結果の画像
- images: 学習データ
- MODEL:学習したモデル構造や重み
- tflog:tensorboardのeventファイル.作成にはmodel.pyのtbを指定．
- weights:n epochごとの重み

## 実験の流れ
autoencoder.pyで分散表現を獲得し，その分散表現を用いてclassify.pyで識別する．
```
$ python autoencoder.py
$ python classify.py
```

## 分散表現を2次元平面で可視化
手法としてt-SNEを用いる．
MODEL以下に分散表現が格納されていることを確認し，以下を実行．
```
$ python t_sne.py
```
figures/t_sne/ 内に画像が保存される．
