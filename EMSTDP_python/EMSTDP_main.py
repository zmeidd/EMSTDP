import os

cpu_num = "2"       # choose number of cores to run the model (CPU)
os.environ["MKL_NUM_THREADS"] = cpu_num
os.environ["MKL_DOMAIN_NUM_THREADS"] = "MKL_BLAS="+cpu_num
os.environ["OMP_NUM_THREADS"] = cpu_num
os.environ["MKL_DYNAMIC"] = "False"
os.environ["OMP_DYNAMIC"] = "False"

import sys
# import pyximport; pyximport.install()

import numpy as np
import EMSTDP_algo as svp
# import BPSNN_allSpikeErr_batcheventstdp2 as svp
import matplotlib.pyplot as plt
import pickle
import gzip
import time
import random

from tqdm import tqdm, tqdm_gui, tnrange, tgrange, trange
# import progressbar


## Import MNIST dataset (to load other dataset, format it to the MNIST format in Keras)

from keras.datasets import mnist                        # load MNIST dataset
# from keras.datasets import fashion_mnist as mnist     # load Fashion MNIST dataset

all_data = mnist.load_data()
TRAINING = all_data[0]
TESTING = all_data[1]

_, h, w = np.shape(TRAINING[0])

total_train_size = np.size(TRAINING[1])     # total number of training samples
total_test_size = np.size(TESTING[1])       # total number of testing samples

train_size = 10000                          # verification period
test_size = 10000                           # verification test size
ver_period = train_size

epochs = 1000                              # number of epochs


## extract images and labels
data = np.expand_dims(np.reshape(TRAINING[0][0:total_train_size], [total_train_size, h*w]), axis=0).astype(float)/255.0
label = np.zeros((total_train_size, 10))
for i in range(total_train_size):
    label[i, TRAINING[1][i]] = 1
label = np.argmax(label, axis=1).astype(int)

dataTest = np.expand_dims(np.reshape(TESTING[0][0:total_test_size], [total_test_size, h*w]), axis=0).astype(float)/255.0
labelTest = np.zeros((total_test_size, 10))
for i in range(total_test_size):
    labelTest[i, TESTING[1][i]] = 1
labelTest = np.argmax(labelTest, axis=1).astype(int)

data_index = (np.linspace(0, total_train_size - 1, total_train_size)).astype(int)


# initialize hyper-parameters (the descriptions are in the Network class)
h = [100]  # [100,300,500,700,test_size,1500]

ind = -1

T = 100
twin = int( T / 2 - 1)
epsilon = 3
scale = 1.0
bias = 0.0
batch_size = 10
tbs = 1000
fr = 1.0
rel = 0
delt = 5
clp = True
lim = 1.0
dropr = 0.0

final_energy = np.zeros([epochs])

hiddenThr1 = 0.5
outputThr1 = 0.1

energies = np.zeros([train_size])
batch_energy = np.zeros([int(train_size / 50)])  # bach_size = 50
ind += 1
acc = []

tmp_rand = np.random.random([T, 1, 1])
randy = np.tile(tmp_rand, (1, batch_size, 784))
tmp_d = np.zeros([T, batch_size, 784])

lr = 0.001
# def __init__(self, dfa, dropr, evt, norm, rel, delt, dr, init, clp, lim, inputs, hiddens, outputs, threshold_h, threshold_o, T=100, bias=0.0, lr=0.0001, scale=1.0, twin=100, epsilon=2):
snn_network = svp.Network(0, dropr, 0, 0.0, rel, delt, 1, 0, clp, lim, 784, h, 10, hiddenThr1*fr, outputThr1*fr, T, bias, lr, scale, twin, epsilon)

## load trained weights, load_weight=True
load_weight = False
load_epoch = 1
if load_weight == True:
    with open('ray_SNN_W_epoch_42.pickle', 'rb') as f2:
        save2 = pickle.load(f2)
        snn_network.w_h = save2['w_h']
        snn_network.w_o = save2['w_o']

s_index = data_index
# for ep in trange(epochs):
for ep in range(epochs):
    snn_network.lr = 1.0 / (2.0 * (5000.0 + int((ep) / 300)))
    pred1 = np.zeros([train_size])
    # np.random.shuffle(s_index)
    spikes = np.zeros([T, batch_size, 784]).astype(float)
    spikes2 = np.zeros([T, batch_size, 784]).astype(float)
    # for i in trange(train_size / batch_size, leave=False):
    for i in trange(int (train_size / batch_size)):
        if ((i + 1) * batch_size % ver_period == 0):  # 5000
            pred = np.zeros([test_size])
            for i2 in range(test_size / tbs):  # train_size

                tmp_rand = np.random.random([T, 1, 1])
                randy = np.tile(tmp_rand, (1, tbs, 784))

                tmp_d = np.tile(dataTest[:, i2 * tbs:(i2 + 1) * tbs, :], (T, 1, 1))
                spikes2 = randy < (tmp_d * fr)

                pred[i2 * tbs:(i2 + 1) * tbs] = snn_network.Test(spikes2.astype(float), tbs)
            acn = sum(pred == labelTest[:test_size]) / float(test_size)
            print( str(ep) + " test_accuray " + str(acn) + " LR " + str(snn_network.lr))
            acc.append(sum(pred == labelTest[:test_size]) / float(test_size))

        tmp_rand = np.random.random([T, 1, 1])
        randy = np.tile(tmp_rand, (1, batch_size, 784))
        tmp_d = np.tile(data[:, s_index[i * batch_size:(i + 1) * batch_size], :], (T, 1, 1))
        spikes = randy < (tmp_d * fr)

        pred1[i * batch_size:(i + 1) * batch_size], energies[i] = snn_network.Train(spikes.astype(float), (
        label[s_index[i * batch_size:(i + 1) * batch_size]]), batch_size)
        # sys.stdout.flush()
    acn = sum(pred1 == label[s_index[:train_size]]) / float(train_size)
    print(str(ep) + " train_accuray " + str(acn))
    np.save("w_h.npy", snn_network.w_h)
    np.save("w_o.npy",snn_network.w_o)

