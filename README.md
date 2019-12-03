# 学習するまでの手順
1. transparent_selected_parts.pyで学習するための画像を作成
2. autoencoder.pyで重みつきモデルを学習
3. classify.pyでクラス識別

##folders
- decoded_imgs:エンコードされた画像とでコードされた画像を表示
- distributed:学習した重みをnumpy形式で保存
- figures:lossとt-SNEの結果の画像
- imgs_param:パラメータでレイヤー重ね合わせをしたデータセット
- MODEL:学習したモデル構造や重み
- oparate_img:画像操作
- results:accuracyなどのcsvファイル
- tflog:tensorboardのeventファイル
- utils:よく使うもの
- weights:5epochごとの重み
- 学術利用4コマ:crop画像
- img_weight_all:すべてのパーツを用いて学習したオートエンコーダを通して生成された画像

##files
- autoencoder.py:オートエンコーダで自己符号の学習，後に中間層の出力を使う
- classify.py:分散表現をMLPにかけ，重み保存．accuracy, predict, metricsをcsv形式で保存する．
- comic2vec.py:モデルを再構築して分散表現化する
- loader.py:画像を読み込む
- load_by_folder: loaderとほぼ同じ．フォルダごとにまとめたarrayを返す．
- model.py:さまざまなモデル定義をまとめている
- remove.py:不必要なファイル，フォルダを削除したいときに
- t_sne.py:分散表現を2次元に圧縮したのち，可視化
- transparent_selected_parts.py:パラメータをsetting.csvから読み取り，必要なデータセットを作成
- test_weight_all.py:オートエンコーダの重みをすべてに固定したものに目抜き，口抜きの画像を入れるとどうなるかの実験
- num2img.py:インデックスから画像への表示の変換
- dist_avg_plus.py:分散表現の変位の平均をとって足してみる実験
- classify_nfilter.py:フィルター数を変えて識別率を観察する実験

## 実験
exp1_1:設定ALL
exp1_2:設定REMOVE_EYE
exp1_3:設定REMOVE_EYE
exp2:多様度識別