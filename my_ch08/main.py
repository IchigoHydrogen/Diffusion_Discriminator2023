import sys, os
sys.path.append(os.pardir)
import cupy as cp
from my_dataset.dataset_load import load_my_dataset
from deep_convnet import DeepConvNet
from common.trainer import Trainer

# データセットをロード
# Real,Fake共に2,000枚の画像から32x32にそれぞれリサイズしたものを使用したのが、ver4.6のデータセットです。
(x_train, t_train), (x_test, t_test) = load_my_dataset(normalize=False)
x_train = cp.asarray(x_train.reshape(-1,1,32,32))
x_test = cp.asarray(x_test.reshape(-1,1,32,32))

max_epochs = 1

network = DeepConvNet(input_dim=(1,32,32), hidden_size=1000)

# 学習済みの重みパラメーターを読み込む
#network.load_params("trained_01.pkl")

optimizer_type = 'AdaGrad'
optimizer_param = {'lr': 0.001}

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=10,
                  optimizer=optimizer_type, optimizer_param=optimizer_param,
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("trained_01.pkl")

# test_accをCSVとして保存します。
trainer.save_test_acc_to_csv('test_acc.csv')

# グラフを描画
trainer.plot_distribution_graph()
trainer.plot_loss()
trainer.plot_accuracy()

