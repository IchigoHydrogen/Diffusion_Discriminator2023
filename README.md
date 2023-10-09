# Diffusion_Discriminator2023

## 概要
畳み込みニューラルネットワーク(CNN)を用いた、画像分類を行うためのツールです。  
小規模な深層学習ライブラリから実装しています。    
サンプルプログラムでは、AI生成画像と本物の画像を学習・分類します。  

また、CuPyを使ったCUDAコア処理をサポートしています。  

## デモ
ここでは、ライブラリを用いて画像分類を行う一例を紹介しています。  
[Google Colabで実行](https://colab.research.google.com/drive/1BcjiWriGA97ZIkXCfv7BiLF5klc7pSN0?usp=sharing)

このライブラリを使って学習させたモデルの動作デモは、以下のサイトで確認することができます。  
https://imagecheck.streamlit.app/  

## 引用と謝辞
このライブラリは、書籍「ゼロから作るDeepLearning」で紹介されているライブラリを元に作りました。。  
詳細は以下のリンクをご覧ください:  
https://github.com/oreilly-japan/deep-learning-from-scratch

## 機能説明
書籍「ゼロから作るDeepLearning」ではmnistを使って手描き文字の認識を行いますが、このリポジトリには独自のデータセットをロードし、学習する機能があります。(my_datasetディレクトリ)  

## データセットについて
オリジナルのデータセットを作りました。  
解像度は32x32,グレースケールの画像です。    
label=0はAI生成画像(Fake)、label=1は本物の写真(Real)を意味します。
内容としては、  
AI生成画像はStable-Diffusion1.4で私が生成した画像2,000枚と、写真はWeb上から風景写真を2,000枚です。  
これらを32x32にリサイズし、グレースケール化しています。 

## 使い方(ローカル環境)
まずはリポジトリをCloneしてください。MatplotlibやCuPyなどの各種ライブラリが入っていれば、動作するはずです。  
サンプルプログラムの使用例:  
本リポジトリのあるディレクトリに移動し、  
```
cd Diffusion_Discriminator2023/my_ch07  もしくはmy_ch08  
python main.py  
```
で実行できます。  

## ライセンス
本ツールのライセンスはdeep-learning-from-scratchのものに準じます。(MIT License)  
人・財産・その他権利を侵害する目的での使用はご遠慮ください。  
