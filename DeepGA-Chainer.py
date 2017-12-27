import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, initializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from multiprocessing import Pool
import multiprocessing as multi
import numpy as np
import scipy.misc as spm
from copy import deepcopy
from Links import Linear, Convolution2D, Flatten
from time import time
import gym
from tqdm import tqdm

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(4, 32, ksize=8, stride=4, nobias=True, initialW=initializers.HeNormal())
            self.conv2=L.Convolution2D(32, 64, ksize=4, stride=2, nobias=True, initialW=initializers.HeNormal())
            self.conv3=L.Convolution2D(64, 64, ksize=3, stride=1, nobias=True, initialW=initializers.HeNormal())
            self.l1=L.Linear(3136, 512, nobias=True, initialW=initializers.HeNormal())
            self.l2=L.Linear(512, 18, nobias=True)
    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.l1(h))
        y = self.l2(h)
        return y
    def set_genome(self, seed):
        for s in range(len(seed)):
            np.random.seed(seed[s])
            if s == 0:
                self.conv1.W = np.random.normal(0, 1, (32, 4, 8, 8))
                self.conv2.W = np.random.normal(0, 1, (64, 32, 4, 4))
                self.conv3.W = np.random.normal(0, 1, (64, 64, 3, 3))
                self.l1.W = np.random.normal(0, 1, (512, 3136))
                self.l2.W = np.random.normal(0, 1, (18, 512))
            else:
                self.conv1.W += np.random.normal(0, 0.05, (32, 4, 8, 8))
                self.conv2.W += np.random.normal(0, 0.05, (64, 32, 4, 4))
                self.conv3.W += np.random.normal(0, 0.05, (64, 64, 3, 3))
                self.l1.W += np.random.normal(0, 0.05, (512, 3136))
                self.l2.W += np.random.normal(0, 0.05, (18, 512))

def play(seed):
    genome = Model()
    genome.set_genome(seed)
    observation = env.reset()
    pop[p][0] = 0.0
    t = 0
    obs[0][t] = scale_image(observation)
    t += 1
    r = 0
    action = 0
    done = False
    while not done:
        #env.render()
        if t%4 == 0:
            with chainer.no_backprop_mode():
                y = genome(obs)
            action = np.argmax(y.data[0])
        observation, reward, done, info =  env.step(action)
        obs[0][t%4] = scale_image(observation)
        pop[p][0] += reward
        t += 1

def scale_image(observation):
    img = rgb2gray(observation)  # Convert RGB to Grayscale
    return (spm.imresize(img, (110, 84)))[110-84-8:110-8, :]  # Scaling

def rgb2gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

pop_size = 500
pop = [[0,[np.random.randint(0,1000000)]] for i in range(pop_size)]
env = gym.make('Kangaroo-v0')
obs = np.zeros((1,4,84,84)).astype(np.float32)
p = Pool(12)

for generation in range(10000):
    max_score = -100
    for p in tqdm(range(pop_size)):
        genome = Model()
        genome.set_genome(pop[p][1])
        observation = env.reset()
        pop[p][0] = 0.0
        t = 0
        obs[0][t] = scale_image(observation)
        t += 1
        r = 0
        action = 0
        done = False
        while not done:
            #env.render()
            if t%4 == 0:
                with chainer.no_backprop_mode():
                    y = genome(obs)
                action = np.argmax(y.data[0])
            observation, reward, done, info =  env.step(action)
            obs[0][t%4] = scale_image(observation)
            pop[p][0] += reward
            t += 1
        if max_score < pop[p][0]:
            max_score = pop[p][0]
    pop.sort(key=lambda x:x[0])
    pop.reverse()
    print('generation:'+str(generation)+' max_score:'+str(max_score))
    for p in range(10, pop_size):
        r = np.random.randint(0,10)
        pop[p][1] = deepcopy(pop[r][1])
        pop[p][1].append(np.random.randint(0,1000000))