from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os, glob


def load_train_data(train_path,test_path):
    print('load_dataset...')
    train_folders = os.listdir(train_path)
    test_folders = os.listdir(test_path)
    train_x = []
    test_x = []
    train_y = []
    test_y = []

    for index, fol_name in enumerate(train_folders): #fol_nameが1の時、indexは0
        files = glob.glob(train_path + '/' + fol_name + '/*.jpg') #フォルダの中の画像ファイルを変数に入れる(リスト形式)
        for file in files:
            image = Image.open(file) #ファイルを開く
            data = np.asarray(image).transpose(2,0,1) #imageと同期する変数を作る
            train_x.append(data)
            train_y.append(index) 
            
    for index, fol_name in enumerate(test_folders):
        files = glob.glob(test_path + '/' + fol_name + '/*.jpg')
        for file in files:
            image = Image.open(file)
            data = np.asarray(image).transpose(2,0,1)
            test_x.append(data)
            test_y.append(index)
        
    train_X = np.array(train_x)
    test_X = np.array(test_x)
    train_Y = np.array(train_y)
    test_Y = np.array(test_y)
    
    oh_encoder = OneHotEncoder(categories='auto', sparse=False)
    train_oh = oh_encoder.fit_transform(pd.DataFrame(train_Y))
    test_oh = oh_encoder.fit_transform(pd.DataFrame(test_Y))
    X_train = train_X
    X_test = test_X
    Y_train = train_oh
    Y_test = test_oh
    return X_train, X_test, Y_train, Y_test

def main(train_path, test_path):
    x_train, x_test, y_train, y_test = load_train_data(train_path,test_path)
        
    return x_train, x_test, y_train, y_test


# train_path = './images/dataset/train_da/train(36,24)_da'
# test_path = './images/dataset/test_da/test(36,24)_da'
# x_train, x_test, t_train, t_test = main(train_path, test_path) 
# print('X_train',x_train.shape) #(5505,3,36,24)
# print('X_test', x_test.shape)  #(1843,3,36,24)
# print('Y_train',t_train.shape) #(5505,37)
# print('Y_test', t_test.shape)  #(1843,37)



    
    


    
    

    