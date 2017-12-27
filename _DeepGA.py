import numpy as np
from numpy.random import *
import scipy.misc as spm
from copy import deepcopy
from Links import Linear, Convolution2D, Flatten
from time import time
import gym

class Genome():
    def __init__(self):
        self.genome = np.zeros((1, 32*8*8 + 64*4*4 + 64*3*3 + 3136*512 + 512*19))
        self.model = []
        self.model.append(Convolution2D(4,32,8,4))
        self.model.append(Convolution2D(32,64,4,2))
        self.model.append(Convolution2D(64,64,3,1))
        self.model.append(Flatten())
        #self.model.append(3136,19)
        self.model.append(Linear(3136,512))
        self.model.append(Linear(512,19))
    def __call__(self, x):
        for model in self.model:
            x = model(x)
        return x
    def set_genome(self, g=None):
        if g is not None:
            self.genome = g
        start = 0
        #print(self.genome.shape)
        self.model[0].filter = np.reshape(self.genome[0,0:32*8*8],(32,8,8))
        start += 32*8*8
        self.model[1].filter = np.reshape(self.genome[0,start:start+64*4*4],(64,4,4))
        start += 64*4*4
        self.model[2].filter = np.reshape(self.genome[0,start:start+64*3*3],(64,3,3))
        start += 64*3*3
        self.model[4].W = np.reshape(self.genome[0,start:start+3136*512],(1,-1))
        start += 3136*512
        self.model[5].W = np.reshape(self.genome[0,start:start+512*19],(1,-1))
    def get_genome(self):
        start = 0
        self.genome[0,0:32*8*8] = np.reshape(self.model[0].filter,(1,-1))[0]
        start += 32*8*8
        self.genome[0,start:start+64*4*4] = np.reshape(self.model[1].filter,(1,-1))[0]
        start += 64*4*4
        self.genome[0,start:start+64*3*3] = np.reshape(self.model[2].filter,(1,-1))[0]
        start += 64*3*3
        self.genome[0,start:start+3136*512] = np.reshape(self.model[4].W,(1,-1))[0]
        start += 3136*512
        #print(np.reshape(self.model[5].W,(1,-1)).shape)
        self.genome[0,start:start+512*19] = np.reshape(self.model[5].W,(1,-1))[0]
        return self.genome

def scale_image(observation):
    img = rgb2gray(observation)  # Convert RGB to Grayscale
    return (spm.imresize(img, (110, 84)))[110-84-8:110-8, :]  # Scaling

def rgb2gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

pop = [[randint(0,1000000)] for i in range(500)]
env = gym.make('Kangaroo-v0')
obs = np.zeros((4,84,84))

done = False
for p in pop:
    seed(p[0])
    genome = Genome()
    observation = env.reset()
    t = 0
    obs[t] = scale_image(observation)
    t += 1
    action = 0
    while not done:
        env.render()
        if t%4 == 0:
            y = genome(obs)
            action = np.argmax(y[0])
        observation, reward, done, info =  env.step(action)
        obs[t%4] = scale_image(observation)
        t += 1