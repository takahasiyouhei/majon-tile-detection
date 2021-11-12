import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from CNN.My_ConvNet import SimpleConvNet
from common.Mytrainer import Trainer
from my_load_data import *

train_path = './images/dataset/train_da/train(36,24)_da'
test_path = './images/dataset/test_da/test(36,24)_da'
x_train, x_test, t_train, t_test = main(train_path, test_path)


max_epochs = 20

network = SimpleConvNet(input_dim=(3,36,24),
                        conv_param={'filter_num':30, 'filter_size':5,'pad':2,'stride':1},
                        hidden_size=100, output_size=37, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_params={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)

trainer.train()


network.save_params("params.pkl")
print('Saved Netwaork Parameters!')





