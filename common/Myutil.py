from sys import _xoptions
import numpy as np
from numpy.core.records import fromarrays

def smooth_curve(x):
    """損失関数のグラフをなめらかにするために用いる
    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]] #スライス表記で多次元配列を作成できる,左は3つの引数を結合している
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid') #二つの変数でたたみ込み積分を行う
    return y[5:len(y)-5]


def shuffle_dataset(x,t):
    """データセットのシャッフルを行う
    Parameters
    ----------
    x : 訓練データ
    y : 教師データ
    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:] #xがどんなデータ型で入力されているかに注目
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    """入力画像サイズ、フィルターのサイズ、ストライド、パディングを考慮した出力数を算出
    """
    return (input_size + 2*pad - filter_size) / stride + 1 

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    parameters
    ----------
    input_data : (データ数,チャンネル数,高さ,幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad :パディング
    Returns
    -------
    col : 2次元配列
    """
    N,C,H,W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1   #//は除算(端数切り捨て)
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) #メモリを確保(6次元配列) 

    for y in range(filter_h): #yはフィルターの高さを取得
        y_max = y + stride*out_h 
        for x in range(filter_w): #xはフィルターの幅を順に取得
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride] #スライスの3つ目は飛ばす数,スライス表記を入れても次元数は変わらない

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
    #返戻値はimgをフィルターと演算しやすいように変形した2次元配列

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    prameters
    ---------
    col : 2次元配列
    input_shape : 入力データの形状(例：(10, 1, 28, 28)) MNISTの場合
    filter_h :
    filter_w :
    stride :
    pad :
    Returns :
    ---------
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    #colを(N, C, filter_h, filter_w, out_h, out_w)の6次元配列に並び替え
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
 

