# Diffusion_Discriminator2023

https://imagecheck.streamlit.app/
このサイトで使っているモデルの学習用ライブラリ。

書籍「ゼロから作るDeepLearning」で紹介されているライブラリを元に、CUDAコアに対応させました。(CuPy)
https://github.com/oreilly-japan/deep-learning-from-scratch

また、Mnistではなく独自データセットをロードして学習できるよう、.pyファイルを追加しました。
それらはmyself_datasetにあります。
サンプルとして、20230523_3rd_28x28_datasetを入れました。label=0がAI生成画像(Fake),label=1が本物の写真です。(Real)

このツールのライセンス:
deep-learning-from-scratch-LICENSEに準じます。
