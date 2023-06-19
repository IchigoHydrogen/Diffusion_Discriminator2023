# coding: utf-8
#cupyに修正済
import cupy as cp


def identity_function(x):
    return x


def step_function(x):
    return cp.array(x > 0, dtype=cp.int)


def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return cp.maximum(0, x)


def relu_grad(x):
    grad = cp.zeros_like(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    x = x - cp.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return cp.exp(x) / cp.sum(cp.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y, t):
    return 0.5 * cp.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
