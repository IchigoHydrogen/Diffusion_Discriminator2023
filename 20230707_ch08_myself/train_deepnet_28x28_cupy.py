import sys, os
sys.path.append(os.pardir)
import cupy as cp
from myself_data.dataset_3rd_28x28 import load_3rd_28x28_dataset
from deep_convnet import DeepConvNet
from common.trainer import Trainer

# データセットをロード
(x_train, t_train), (x_test, t_test) = load_3rd_28x28_dataset(normalize=False, one_hot_label=True)
x_train = cp.asarray(x_train.reshape(-1,1,28,28))
x_test = cp.asarray(x_test.reshape(-1,1,28,28))

max_epochs = 50

network = DeepConvNet(input_dim=(1,28,28), hidden_size=500)

# 学習済みの重みパラメーターを読み込む
#network.load_params("deep_convnet_params_original_1.pkl")
#print("Loaded Network Parameters!")

optimizer_type = 'AdaGrad'
optimizer_param = {'lr': 0.001}

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=10,
                  optimizer=optimizer_type, optimizer_param=optimizer_param,
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("3rd_28x28_AdaGrad_Epoch50_hidden500.pkl")
#print("Saved Network Parameters!")

# test_accをCSVとして保存します。
trainer.save_test_acc_to_csv('test_acc.csv')

#グラフを描画
trainer.plot_distribution_graph()
trainer.plot_loss()
trainer.plot_accuracy()

