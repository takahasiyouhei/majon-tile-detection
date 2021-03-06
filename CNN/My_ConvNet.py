import pickle
import numpy as np
from collections import OrderedDict

from numpy.lib.function_base import _gradient_dispatcher #項目が追加された順序を記憶する辞書のサブクラス
from common.Mylayers import *
from common.Myfunctions import softmax
from common.Mytrainer import Trainer


class SimpleConvNet:
    """単純なConvNet
    conv - relu - pool - conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size : 入力サイズ(MNISTの場合は1channel(グレスケ)　×28pixel×28pixel)
    hidden_size_list : 隠れ層のニューロンの数のリスト (e.g. [100, 100, 100])
    output_size : 出力サイズ(MNISTの場合は10)
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定(e.g. 0.01)
        'relu'または'He'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「xavierの初期値」を設定
    """

    def __init__(self, input_dim = (3, 36, 24),
                conv_param={'filter_num':30, 'filter_size':5, 'pad':2, 'stride':1},
                hidden_size=100, output_size=37, weight_init_std=0.01):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_h = input_dim[1]
        input_w = input_dim[2]
        conv1_output_h = (input_h - filter_size + 2*filter_pad) / filter_stride + 1
        conv1_output_w = (input_w - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_h = int(conv1_output_h/2)
        pool_output_w = int(conv1_output_w/2)
        conv2_output_h = (pool_output_h - filter_size + 2*filter_pad) / filter_stride + 1
        conv2_output_w = (pool_output_w - filter_size + 2*filter_pad) / filter_stride + 1
        # (2, 2)のmax pooling後の画素数の合計値(affineレイヤーの重み行列のサイズ用)
        pool2_output_size = int(filter_num * (conv2_output_h/2) * (conv2_output_w/2))
        input_size = input_dim[1]

        #重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(filter_num, filter_num, filter_size, filter_size)
        self.params['b2'] = np.zeros(filter_num)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(pool2_output_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        #レイヤの生成
        self.layers = OrderedDict() #追加したレイヤの順番を記憶(逆伝播の際に用いる)
        self.layers['Conv1'] = Convolution(self.params['W1'],self.params['b1'],
                                            conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                            conv_param['stride'], conv_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss() #予測時にsoftmax関数は使わないので、self.layerとは別の変数とする

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=1):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        '''勾配を求める(誤差逆伝播法)
        parameters
        ----------
        x : 入力データ
        t : 教師ラベル
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1'], grads['W2'], ･･･は各層の重み
            grads['b1'], grads['b2'], ･･･は各層のバイアス
        '''
        #　forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout) #最後のレイヤだけは別枠で処理

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name='params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Affine1', 'Affine2']): #オブジェクトにインデックスを付与
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]












