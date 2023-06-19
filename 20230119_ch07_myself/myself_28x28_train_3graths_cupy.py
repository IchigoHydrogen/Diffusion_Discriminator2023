#cupyに変更済み
import sys, os
sys.path.append(os.pardir)
import cupy as cp
from myself_data.dataset_3rd_28x28 import load_3rd_28x28_dataset
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

#データセットをロード
(x_train, t_train), (x_test, t_test) = load_3rd_28x28_dataset(normalize=True, one_hot_label=True)
x_train = cp.asarray(x_train.reshape(-1,1,28,28))
x_test = cp.asarray(x_test.reshape(-1,1,28,28))

#ここでepoch数を指定
max_epochs = 3

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=2, weight_init_std=0.01)

#重みパラメータをロード
#network.load_params("params.pkl")
#print("Loaded Network Parameters!")

#手法はSGD,Momentum,Nesterov,AdaGrad,RMSprop,Adamの6種類。
optimizer_type = 'Adam'
optimizer_param = {'lr': 0.001}

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer=optimizer_type, optimizer_param=optimizer_param,
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

#パラメータの保存
#network.save_params("params.pkl")
#print("Saved Network Parameters!")

#グラフを描画
trainer.plot_distribution_graph()
trainer.plot_loss()
trainer.plot_accuracy()
