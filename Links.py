import numpy as np
from numpy.random import *
import math

class Linear:
    def __init__(self, inp_n, out_n, activation=None, initialization=None):
        if initialization == "HeNormal":
            self.W = np.r_[normal(0, np.sqrt(2./inp_n), (inp_n, out_n)), np.zeros((1,out_n))]
        elif initialization == "zeros":
            self.W = np.zeros((inp_n+1,out_n))
        else:
            self.W = np.r_[normal(0, np.sqrt(1./inp_n), (inp_n, out_n)), np.zeros((1,out_n))]
        self.x = None
        self.activation = activation
    def __call__(self, inp):
        inp = np.c_[inp,np.ones((1,1))]
        y = np.dot(inp, self.W)
        if self.activation == "sigmoid":
            return 1. / (1 + np.exp(-1*y))
        elif self.activation == "tanh":
            return np.tanh(y)
        elif self.activation == "relu":
            return y * (y > 0)
        return y

class Convolution2D:
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, nobias=True, initialW=None, activation= None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.filter = normal(0, 1.0, (out_channnels, ksize, ksize))
        self.activation = activation
    def __call__(self, x):
        y = np.zeros((self.out_channels, math.floor(x[0] - self.ksize)/self.stride), math.floor(x[0] - self.ksize)/self.stride))
        for in_c in x:
            for out_c in y:
                

