import numpy as np
import matplotlib.pyplot as plt


path = './images/dataset/test/1/1_0.png'
C = 3
H = 36
W = 24
a = np.arange(C*H*W).reshape(C,H,W)
b = a.transpose(1,2,0)
print(a[2][23][:].shape)
print(b[23][:][1].shape)

