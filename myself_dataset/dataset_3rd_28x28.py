import os
import numpy as np
from PIL import Image

# 画像の保存先のディレクトリ
save_dir='C:/Users/ei/Desktop/Program/deep-learning-from-scratch1-myself-deepfake_cupy/deep-learning-from-scratch1-myself-deepfake_cupy/myself_data/20230523_3rd_28x28_dataset'
# 画像の読み込み元のディレクトリ
image_dir ='C:/Users/ei/Desktop/Program/deep-learning-from-scratch1-myself-deepfake_cupy/deep-learning-from-scratch1-myself-deepfake_cupy/myself_data/20230523_3rd_28x28_dataset'

# 画像を保存するためのNumpy配列
x_train = []
x_test = []

# ラベルを保存するためのNumpy配列
y_train = []
y_test = []

classes = ['Fake','Real']

# 画像を読み込んで、特定の解像度にリサイズし、Numpy配列に変換する。png以外のファイルは読み込まない
def load_image(image_dir, classes):
    for index, classlabel in enumerate(classes):
        photos_dir = image_dir + '/' + classlabel
        files = os.listdir(photos_dir)
        for i, file in enumerate(files):
            if file.endswith('.png'):
                # 画像を開く
                image = Image.open(photos_dir + '/' + file)
                # 画像を32x32にリサイズ
                image=image.resize((28,28))
                #画像をグレースケール化。この1行で全てのエラーが解決するんだ...
                image = image.convert('L')
                # 画像をNumpy配列に変換
                image_data = np.asarray(image)
                
                # テスト用データセットに追加
                if i %10 == 0:
                    x_test.append(image_data)
                    y_test.append(index)
                # 訓練用データセットに追加
                else:
                    x_train.append(image_data)
                    y_train.append(index)

# 画像を読み込んで、28x28にリサイズし、Numpy配列に変換する
"""

load_image(image_dir, classes)

# 訓練用データセットを保存
x_train = np.array(x_train)
y_train = np.array(y_train)
np.save(save_dir + '/x_train.npy', x_train)
np.save(save_dir + '/y_train.npy', y_train)


# テスト用データセットを保存
x_test = np.array(x_test)
y_test = np.array(y_test)
np.save(save_dir + '/x_test.npy', x_test)
np.save(save_dir + '/y_test.npy', y_test)

#ちゃんと実行されなくても言ってくれるよ。 """
print('this dataset program has done!!')

# データセットをロードする関数
def load_3rd_28x28_dataset(normalize=True,flatten=True,one_hot_label=False):
    # 訓練用データセットをロード
    x_train = np.load('C:/Users/ei/Desktop/Program/deep-learning-from-scratch1-myself-deepfake_cupy/deep-learning-from-scratch1-myself-deepfake_cupy/myself_data/20230523_3rd_28x28_dataset/x_train.npy')
    y_train = np.load('C:/Users/ei/Desktop/Program/deep-learning-from-scratch1-myself-deepfake_cupy/deep-learning-from-scratch1-myself-deepfake_cupy/myself_data/20230523_3rd_28x28_dataset/y_train.npy')
    # テスト用データセットをロード
    x_test = np.load('C:/Users/ei/Desktop/Program/deep-learning-from-scratch1-myself-deepfake_cupy/deep-learning-from-scratch1-myself-deepfake_cupy/myself_data/20230523_3rd_28x28_dataset/x_test.npy')
    y_test = np.load('C:/Users/ei/Desktop/Program/deep-learning-from-scratch1-myself-deepfake_cupy/deep-learning-from-scratch1-myself-deepfake_cupy/myself_data/20230523_3rd_28x28_dataset/y_test.npy')

    if normalize==True:
        # x_train, x_test を float32 型に変換
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        
        x_train /= 255.0
        print("x_train normalized")

        x_test /= 255.0
        print("x_test normalized")

    return (x_train, y_train), (x_test, y_test)