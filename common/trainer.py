# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
from common.optimizer import *

class Trainer:
    #ニューラルネットの訓練を行うクラス
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 #epochs=20, mini_batch_size=100,
                 epochs=20, mini_batch_size=10,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = cp.array(x_train)
        self.t_train = cp.array(t_train)
        self.x_test = cp.array(x_test)
        self.t_test = cp.array(t_test)
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = self.x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        print("this is self.current_iter",self.current_iter)
        print("this is self.iter_per_epoch",self.iter_per_epoch)
        print("this is self.current_epoch",self.current_epoch)

        batch_mask = cp.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

            #最高精度のエポックと精度の数値を出力する
            max_test_acc = max(self.test_acc_list)
            max_test_acc_epoch = self.test_acc_list.index(max_test_acc)
            print("Highest accuracy reached at epoch:", max_test_acc_epoch)
            print("Highest test accuracy:", max_test_acc)

    #以下に追加した関数

    #配列用のsoftmax関数
    def array_softmax(self, x):
        exp_x = cp.exp(x)
        sum_exp_x = cp.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp_x

    #テストデータの判別確率の分布グラフを描画する関数
    def plot_distribution_graph(self):
        # Verify test data
        label_percentages = self.network.predict(self.x_test)

        probabilities = self.array_softmax(label_percentages)
        probabilities = cp.asnumpy(probabilities)

        probabilities_label_0 = probabilities[cp.asnumpy(self.t_test) == 0]
        probabilities_label_1 = probabilities[cp.asnumpy(self.t_test) == 1]

        data_0 = {'dp_0': probabilities_label_0[:, 0],
                  'dp_1': probabilities_label_0[:, 1]}
        df_0 = pd.DataFrame(data_0)

        data_1 = {'dp_0': probabilities_label_1[:, 0],
                  'dp_1': probabilities_label_1[:, 1]}
        df_1 = pd.DataFrame(data_1)

        df_0.to_csv('output_label_0.csv', index=False)
        df_1.to_csv('output_label_1.csv', index=False)

        # Draw graph
        plt.figure(figsize=(12, 6))
        plt.rcParams['font.family'] = "MS Gothic"
        plt.rcParams['font.weight'] = 'bold'
        plt.xlim(0.0,1.0)
        plt.hist(df_0['dp_0'], bins=20, alpha=0.5, label='Fake Image')
        plt.hist(df_1['dp_0'], bins=20, alpha=0.5, label='Real Image')
        plt.xlabel('確率',fontname="MS Gothic")
        plt.ylabel('頻度',fontname="MS Gothic")
        plt.title("AI生成画像と判断した確率", fontsize=16, fontname="MS Gothic")
        plt.legend()

        # Insert max_epochs text into Plot 1
        max_epochs_text = f'max_epochs={self.epochs}'
        plt.text(0.1, 0.9, max_epochs_text, fontsize=14, transform=plt.gca().transAxes) # coordinate values are scaled to axes

        plt.show()

    #損失の度グラフを描画する関数
    def plot_loss(self):
        plt.figure()
        x = cp.asnumpy(cp.arange(len(self.train_loss_list)))
        y1 = cp.asnumpy(cp.array(self.train_loss_list))
        plt.plot(x, y1, label='train')
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("Loss per iteration")
        plt.legend()
        plt.show()

    #テストデータの精度グラフを描画する関数
    def plot_accuracy(self):
        plt.figure()
        x = cp.asnumpy(cp.arange(len(self.train_acc_list)))
        #y1 = cp.asnumpy(cp.array(self.train_acc_list))
        y2 = cp.asnumpy(cp.array(self.test_acc_list))
        #plt.plot(x, y1, label='train')
        plt.plot(x, y2, label='test', linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.title("Accuracy per epoch")
        plt.legend()
        plt.show()