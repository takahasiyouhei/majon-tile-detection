import numpy as np

def identity_function(x):
    return x

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    #シグモイド関数

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0, x)
    #分岐させなくても大丈夫

def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    #配列に入れるのはなぜ?

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0) #データの最大値を取得してオーバーフロー対策(axisの指定がないとすべての中から1つになる)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
    #自乗誤差を返す関数

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) 
        y = y.reshape(1, y.size)

    #教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size: #one-hot以外はインデックスで指定されている 
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)