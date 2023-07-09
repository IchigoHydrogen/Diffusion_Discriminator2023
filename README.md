# Diffusion_Discriminator2023

## 概要
これはシンプルな畳み込みニューラルネットワークを用いた、画像分類を行うための機械学習ライブラリです。  
サンプルプログラムでは、AI生成画像と本物の画像を学習・分類します。  

また、CuPyを使ったCUDAコア処理に対応しています。

## デモ
このライブラリを使って学習させたモデルの動作デモは、以下のサイトで確認することができます。  
https://imagecheck.streamlit.app/  

## 機能説明
mnistではなく独自のデータセットをロードし、学習する機能があります。(my_datasetディレクトリ)  
サンプルデータセットを用意しました。  
label=0はAI生成画像(Fake)、label=1は本物の写真(Real)を意味します。  
512x512の画像計4,000枚を、それぞれ32x32にリサイズしたものです。  

## 使い方  
まずはリポジトリをクローンしてください。MatplotlibやCuPyなど、各種ライブラリが入っていれば、動作するはずです。  
サンプルプログラムの使用例:  
cd Diffusion_Discriminator2023/my_ch07  
python main.py  
で実行できます。  


## 引用と謝辞
このライブラリは、書籍「ゼロから作るDeepLearning」で紹介されているライブラリを元に作られました。  
詳細は以下のリンクをご覧ください:  
https://github.com/oreilly-japan/deep-learning-from-scratch


## ライセンス
本ツールのライセンスはdeep-learning-from-scratchのものに準じます。(MIT License)  
人・財産・その他権利を侵害する目的での使用はご遠慮ください。  
