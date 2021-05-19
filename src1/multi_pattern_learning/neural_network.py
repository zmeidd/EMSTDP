# encoding=utf8
# -*- coding: utf-8 -*-
"""
2 layers Neural network applied to handwriting recognition
from MNIST database.
"""

from __future__ import division
import time
import pickle
import gzip
from random import randint
from scipy import misc
from scipy import special
import numpy as np

import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["MKL_DOMAIN_NUM_THREADS"] = "MKL_BLAS=1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_DYNAMIC"] = "False"
# os.environ["OMP_DYNAMIC"] = "False"
# =====================
#    Initialisation
# =====================

# # Initialisation - Import from MNIST database
# START_TIME = time.time()
# ft = gzip.open('data_training', 'rb')
# TRAINING = pickle.load(ft)
# ft.close()
# ft = gzip.open('data_testing', 'rb')
# TESTING = pickle.load(ft)
# ft.close()

# print('Import duration '+str(round((time.time() - START_TIME), 2))+'s')
# print('----')

import random
import numpy as np
import collections
import pickle
import sys
import os
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

# import .computeResults as compResults
# from .computeResults import computeResults


def readMSTARFile(filename):
    # raw_input('Enter the mstar file to read: ')

    # print filename

    f = open(filename, 'rb')

    a = b''

    phoenix_header = []

    tmp = 'PhoenixHeaderVer'
    while tmp.encode() not in a:
        a = f.readline()

    a = f.readline()

    tmp = 'EndofPhoenixHeader'
    while tmp.encode() not in a:
        phoenix_header.append(a)
        a = f.readline()

    data = np.fromfile(f, dtype='>f4')

    # print data.shape

    # magdata = data[:128*128]
    # phasedata = data[128*128:]

    # if you want to print an image
    # imdata = magdata*255

    # imdata = imdata.astype('uint8')

    targetSerNum = '-'

    for line in phoenix_header:
        # print line
        if ('TargetType').encode() in line:
            targetType = line.strip().split(b'=')[1].strip()
        elif ('TargetSerNum').encode() in line:
            targetSerNum = line.strip().split(b'=')[1].strip()
        elif ('NumberOfColumns').encode() in line:
            cols = int(line.strip().split(b'=')[1].strip())
        elif ('NumberOfRows').encode() in line:
            rows = int(line.strip().split(b'=')[1].strip())

    label = targetType  # + '_' + targetSerNum

    roffset = (rows - 128) // 2
    coffset = (cols - 128) // 2
    data = data[:rows * cols]
    data = data.reshape((rows, cols))
    data = data[roffset:(128 + roffset), coffset:(128 + coffset)]

    # plt.imshow(data)
    # plt.show()

    return data.astype('float32'), label, targetSerNum


def readMSTARDir(dirname, ww, hh):
    data = np.zeros([0, ww, hh], dtype='int')
    labels = []
    serNums = []
    files = os.listdir(dirname)

    for f in files:
        fullpath = os.path.join(dirname, f)
        if os.path.isdir(fullpath):
            if 'SLICY' in f:
                continue
            d, l, sn = readMSTARDir(fullpath, ww, hh)
            data = np.concatenate((data, d), axis=0)
            labels = labels + l
            serNums = serNums + sn
        else:
            # print
            # fullpath
            if not fnmatch(f, '*.[0-9][0-9][0-9]'):
                continue
            d, l, sn = readMSTARFile(os.path.join(dirname, f))
            # print
            dd = d.shape

            # d = d[int(dd[0]/2-ww/2):int(dd[0]/2+ww/2), int(dd[1]/2-hh/2):int(dd[1]/2+hh/2)]

            d = d[int(dd[0] / 2 - ww):int(dd[0] / 2 + ww), int(dd[1] / 2 - hh):int(dd[1] / 2 + hh)]
            d = resize(d, (ww, hh), anti_aliasing=True)

            dmx = np.max(d)
            dmn = np.min(d)
            d = ((d-dmn)/(dmx-dmn))*255

            # d[d < 25] = 0
            # d[d>50] = d[d>50]*2
            # d[d > 255] = 255
            d = d.astype(int)
            # d = ((d - dmn) / (dmx - dmn))
            # plt.imshow(d)
            # plt.show()

            # if l == ('t72_tank').encode():
            #     print("T72")
            # if l == ('bmp2_tank').encode():
            #     print("T72")
            # if l == ('btr70_transport').encode():
            #     print("T72")

            data = np.concatenate((data, d.reshape(1, ww, hh)), axis=0)
            labels = labels + [l]
            serNums = serNums + [sn]

    return data, labels, serNums


# filename = sys.argv[1]
# outputfile = sys.argv[2]

# dset = "mnist"
#
# if dset == "mnist":
#     from keras.datasets import mnist
#
#     all_data = mnist.load_data()
#     TRAINING = all_data[0]
#     TESTING = all_data[1]
#
#     numlabels = 10
#
#
# if dset == "mstar10":
#     filename = "/homes/amarshrestha/projects/test2/EMSTDP/src1/MSTAR_PUBLIC_MIXED_TARGETS_CD1/"
#
#     ww = 32
#     hh = 32
#     data, labels, serNums = readMSTARDir(filename, ww, hh)
#
#     mstar_dic = dict()
#
#     mstar_dic['data'] = data
#     mstar_dic['labels'] = labels
#     mstar_dic['serial numbers'] = serNums
#
#     labels = list(set(labels))
#
#     label_dict = dict()
#
#     for i in range(len(labels)):
#         label_dict[labels[i]] = i
#
#     for i in range(len(mstar_dic['labels'])):
#         mstar_dic['labels'][i] = label_dict[mstar_dic['labels'][i]]
#
#
#     totaldata = []
#
#     totaldata.append(mstar_dic['data'])
#     totaldata.append(np.array(mstar_dic['labels']))
#
#     total_size = np.size(totaldata[1])
#     total_size = 7500
#     arr = np.arange(total_size)
#     np.random.shuffle(arr)
#
#     TRAINING = []
#     TESTING = []
#
#     TRAINING.append(totaldata[0][arr[0:(total_size-1000)]])
#     TRAINING.append(totaldata[1][arr[0:(total_size-1000)]])
#     TESTING.append(totaldata[0][arr[(total_size-1000):total_size]])
#     TESTING.append(totaldata[1][arr[(total_size-1000):total_size]])
#
#     numlabels = 10
#
# if dset == "mstar3":
#     filename = "/homes/amarshrestha/projects/test2/EMSTDP/src1/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/"
#
#     ww = 32
#     hh = 32
#     data, labels, serNums = readMSTARDir(os.path.join(filename, 'TRAIN'), ww, hh)
#
#     mstar_dic_train = dict()
#
#     mstar_dic_train['data'] = data
#     mstar_dic_train['labels'] = labels
#     mstar_dic_train['serial numbers'] = serNums
#
#     data, labels, serNums = readMSTARDir(os.path.join(filename, 'TEST'), ww, hh)
#
#     mstar_dic_test = dict()
#
#     mstar_dic_test['data'] = data
#     mstar_dic_test['labels'] = labels
#     mstar_dic_test['serial numbers'] = serNums
#
#     labels = list(set(labels))
#
#     label_dict = dict()
#
#     for i in range(len(labels)):
#         label_dict[labels[i]] = i
#
#     for i in range(len(mstar_dic_train['labels'])):
#         mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]
#
#     for i in range(len(mstar_dic_test['labels'])):
#         mstar_dic_test['labels'][i] = label_dict[mstar_dic_test['labels'][i]]
#
#     TRAINING = []
#     TESTING = []
#
#     TRAINING.append(mstar_dic_train['data'])
#     TRAINING.append(np.array(mstar_dic_train['labels']))
#     TESTING.append(mstar_dic_test['data'])
#     TESTING.append(np.array(mstar_dic_test['labels']))
#
#     total_train_size = np.size(TRAINING[1])
#     total_test_size = np.size(TESTING[1])
#     ver_period = np.size(TRAINING[1])
#
#     arr = np.arange(total_train_size)
#     np.random.shuffle(arr)
#
#     TRAINING[0] = TRAINING[0][arr, :, :]
#     TRAINING[1] = TRAINING[1][arr]
#
#     arr = np.arange(total_test_size)
#     np.random.shuffle(arr)
#
#     TESTING[0] = TESTING[0][arr, :, :]
#     TESTING[1] = TESTING[1][arr]
#
#     numlabels = 3
#
# _, h, w = np.shape(TRAINING[0])
#
# total_train_size = np.size(TRAINING[1])
# total_test_size = np.size(TESTING[1])
# ver_period = np.size(TRAINING[1])

# TRAINING = []
# TESTING = []
# h = 0
# w = 0
#
# total_train_size = 0
# total_test_size = 0
# ver_period = 0
# numlabels = 0


def getEMSTDPDataSet(dname="mnist"):
    dset = dname

    # global total_train_size
    # global total_test_size
    # global ver_period
    # global numlabels

    TRAINING = []
    TESTING = []

    if dset == "mnist":
        from keras.datasets import mnist

        all_data = mnist.load_data()
        TRAINING = all_data[0]
        TESTING = all_data[1]

        numlabels = 10

    if dset == "mstar10":
        filename = "/homes/amarshrestha/projects/test2/EMSTDP/src1/MSTAR_PUBLIC_MIXED_TARGETS_CD1/"

        ww = 32
        hh = 32
        # data, labels, serNums = readMSTARDir(filename, ww, hh)
        #
        # mstar_dic = dict()
        #
        # mstar_dic['data'] = data
        # mstar_dic['labels'] = labels
        # mstar_dic['serial numbers'] = serNums
        #
        # labels = list(set(labels))
        #
        # label_dict = dict()
        #
        # for i in range(len(labels)):
        #     label_dict[labels[i]] = i
        #
        # for i in range(len(mstar_dic['labels'])):
        #     mstar_dic['labels'][i] = label_dict[mstar_dic['labels'][i]]

        data, labels, serNums = readMSTARDir(os.path.join(filename, '17_DEG'), ww, hh)

        mstar_dic_train = dict()

        mstar_dic_train['data'] = data
        mstar_dic_train['labels'] = labels
        mstar_dic_train['serial numbers'] = serNums

        data, labels, serNums = readMSTARDir(os.path.join(filename, '15_DEG'), ww, hh)

        mstar_dic_test = dict()

        mstar_dic_test['data'] = data
        mstar_dic_test['labels'] = labels
        mstar_dic_test['serial numbers'] = serNums

        labels = list(sorted(set(labels)))

        label_dict = dict()

        for i in range(len(labels)):
            label_dict[labels[i]] = i

        for i in range(len(mstar_dic_train['labels'])):
            mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]

        for i in range(len(mstar_dic_test['labels'])):
            mstar_dic_test['labels'][i] = label_dict[mstar_dic_test['labels'][i]]

        # totaldata = []
        #
        # totaldata.append(mstar_dic_train['data'])
        # totaldata.append(np.array(mstar_dic_train['labels']))

        total_size = np.size(mstar_dic_train['labels'])
        # total_size = 7500
        arr_train = np.arange(total_size)
        np.random.shuffle(arr_train)

        total_size = np.size(mstar_dic_test['labels'])
        # total_size = 7500
        arr_test = np.arange(total_size)
        np.random.shuffle(arr_test)

        TRAINING = []
        TESTING = []

        TRAINING.append(mstar_dic_train['data'][arr_train])
        TRAINING.append(np.array(mstar_dic_train['labels'])[arr_train])
        TESTING.append(mstar_dic_test['data'][arr_test])
        TESTING.append(np.array(mstar_dic_test['labels'])[arr_test])

        numlabels = 10

    if dset == "mstar3":
        filename = "/homes/amarshrestha/projects/test2/EMSTDP/src1/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/"

        ww = 32
        hh = 32
        data, labels, serNums = readMSTARDir(os.path.join(filename, 'TRAIN'), ww, hh)

        mstar_dic_train = dict()

        mstar_dic_train['data'] = data
        mstar_dic_train['labels'] = labels
        mstar_dic_train['serial numbers'] = serNums

        data, labels, serNums = readMSTARDir(os.path.join(filename, 'TEST'), ww, hh)

        mstar_dic_test = dict()

        mstar_dic_test['data'] = data
        mstar_dic_test['labels'] = labels
        mstar_dic_test['serial numbers'] = serNums

        labels = list(set(labels))

        label_dict = dict()

        for i in range(len(labels)):
            label_dict[labels[i]] = i

        for i in range(len(mstar_dic_train['labels'])):
            mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]

        for i in range(len(mstar_dic_test['labels'])):
            mstar_dic_test['labels'][i] = label_dict[mstar_dic_test['labels'][i]]

        TRAINING = []
        TESTING = []

        TRAINING.append(mstar_dic_train['data'])
        TRAINING.append(np.array(mstar_dic_train['labels']))
        TESTING.append(mstar_dic_test['data'])
        TESTING.append(np.array(mstar_dic_test['labels']))

        total_train_size = np.size(TRAINING[1])
        total_test_size = np.size(TESTING[1])
        ver_period = np.size(TRAINING[1])

        arr = np.arange(total_train_size)
        np.random.shuffle(arr)

        TRAINING[0] = TRAINING[0][arr, :, :]
        TRAINING[1] = TRAINING[1][arr]

        arr = np.arange(total_test_size)
        np.random.shuffle(arr)

        TESTING[0] = TESTING[0][arr, :, :]
        TESTING[1] = TESTING[1][arr]

        numlabels = 3

    # h
    # w


    _, h, w = np.shape(TRAINING[0])

    total_train_size = np.size(TRAINING[1])
    total_test_size = np.size(TESTING[1])
    ver_period = np.size(TRAINING[1])

    # trainXdata = (np.reshape(TRAINING[0], [total_train_size, h * w])).astype(int)
    # testXdata = (np.reshape(TESTING[0], [total_test_size, h * w])).astype(int)

    return TRAINING, TESTING, total_train_size, total_test_size, numlabels, h, w

datset = "mstar10"
# datset = "mnist"

TRAINING,TESTING, total_train_size, total_test_size, numlabels, h, w = getEMSTDPDataSet(datset)
# _, TESTING, total_train_size, total_test_size, numlabels, h, w = getEMSTDPDataSet(datset)
# =====================
#     Network class
# =====================

'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

# from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

batch_size = 16
num_classes = 10
epochs = 20

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = TRAINING[0]
y_train = TRAINING[1]
x_test = TESTING[0]
y_test = TESTING[1]

nt = 'conv'

if nt == 'conv':
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, h, w)
        x_test = x_test.reshape(x_test.shape[0], 1, h, w)
        input_shape = (1, h, w)
    else:
        x_train = x_train.reshape(x_train.shape[0], h, w, 1)
        x_test = x_test.reshape(x_test.shape[0], h, w, 1)
        input_shape = (h, w, 1)
else:
    x_train = x_train.reshape(total_train_size, h*w)
    x_test = x_test.reshape(total_test_size, h*w)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# for i in range(10):
a = collections.Counter(y_train)
print(a)

max_sample = 1000
for i in range(10):
    ind = np.where(y_train==i)
    sz = np.sum(y_train==i)
    ind1 = np.random.choice(np.squeeze(ind), size=max_sample - sz)
    x_train = np.concatenate((x_train, x_train[ind1, :]))
    y_train = np.concatenate((y_train, y_train[ind1]))

total_train_size = np.size(y_train)
ind2 = np.arange(total_train_size)
np.random.shuffle(ind2)
x_train = x_train[ind2, :]
y_train = y_train[ind2]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()

if nt == 'conv':
    model.add(Conv2D(16, kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='valid',
                     use_bias=False,
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(8, kernel_size=(3, 3),
                     strides=(2, 2),
                     padding='valid',
                     use_bias=False,
                     activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
else:
    model.add(Dense(100, activation='relu', input_shape=(h*w,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# class Network:
#
#     def __init__(self, num_hidden):
#         self.input_size = h*w
#         self.output_size = 10
#         self.num_hidden = num_hidden
#         self.best = 0.
#         self.same = 0
#
#
#         # Standardize random weights
#         # np.random.seed(0)
#         hidden_layer = np.random.rand(self.num_hidden, self.input_size) / self.num_hidden
#         output_layer = np.random.rand(self.output_size, self.num_hidden) / self.output_size
#         self.layers = [hidden_layer, output_layer]
#         hidden_a = np.zeros([self.num_hidden])
#         output_a = np.zeros([self.output_size])
#         self.act = [hidden_a, output_a]
#         self.iteration = 0.
#
#         print('Initialization with random weight')
#         print('-----')
#
#     def train(self, batchsize, training):
#         start_time = time.time()
#         print('Network training with '+str(batchsize)+' examples')
#         print('Until convergence (10 iterations without improvements)')
#         print('-----')
#         # inputs = training[0][0:batchsize]
#         targets1 = np.zeros((60000, 10))
#         for i in range(60000):
#             targets1[i, training[1][i]] = 1
#
#         data_index = (np.linspace(0, 60000 - 1, 60000)).astype(int)
#         sidx = data_index
#         per = []
#         # Performs iterations
#         while self.iteration < 100:
#             np.random.shuffle(sidx)
#             inputs = training[0][sidx[0:10000]]
#             targets = targets1[sidx[0:10000], :]
#             for input_vector, target_vector in zip(inputs, targets):
#                 self.backpropagate(input_vector, target_vector)
#             # Messages and backups
#             self.iteration += 1.
#             accu = self.accu(TESTING)
#             message = 'Iteration '+str(int(self.iteration)).zfill(2) + \
#                 ' (' + str(round(time.time()-start_time)).zfill(2)+'s) '
#             message += 'Precision G:'+str(accu[1]).zfill(4)+'% Min:'+ \
#                 str(accu[0]).zfill(4)+ '% ('+str(int(accu[2]))+')'
#             # if accu[1] > self.best:
#             #     self.same = 0
#             #     self.best = accu[1]
#             #     message += ' R'
#             #     if accu[0] > 97:
#             #         self.sauv(file_name='ntMIN_'+str(accu))
#             #         message += 'S'
#             # else:
#             #     self.same += 1
#             per.append(accu[1])
#             print(message)
#         # Final message
#         print('10 Iterations without improvements.')
#         print('Total duration: ' + str(round((time.time() - start_time), 2))+'s')
#
#     def feed_forward(self, input_vector, deltas):
#         """Takes a network (Matrix list) and returns the outputs of both
#          layers by propagating the entry"""
#         outputs = []
#         ind = 0
#         for layer in self.layers:
#             # input_with_bias = np.append(input_vector, 1)   # Ajout constante
#             output = np.dot(layer, input_vector) + deltas[ind]
#             ind += 1
#             output = special.expit(output)
#             outputs.append(output)
#             # The output is the input of the next layer
#             input_vector = output
#         return outputs
#
#     def backpropagate(self, input_vector, target):
#         """Reduce error for one input vector:
#         Calculating the partial derivatives for each coeff then subtracts"""
#         # Calculation of partial derivatives for the output layer and subtraction
#         deltas = [np.zeros([self.num_hidden]), np.zeros([self.output_size])]
#         c = 1. / (int(self.iteration/6)  + 10)  # Learning coefficient
#         # c = 0.01
#         hidden_outputs, outputs = self.feed_forward(input_vector, deltas)
#         # hidden_outputs = self.act[0]
#         output_deltas = -(outputs - target)
#
#         # Calculation of partial derivatives for the hidden layer and subtraction
#         hidden_deltas = np.dot(self.layers[-1].T, output_deltas)
#
#         deltas = [hidden_deltas, output_deltas]
#
#         hidden_outputs1, outputs1 = self.feed_forward(input_vector, deltas)
#
#         self.layers[-1] += c * np.outer(outputs1 - outputs, hidden_outputs)
#         self.layers[0] += c * np.outer(hidden_outputs1 - hidden_outputs, input_vector)
#
#         # self.act[0] =
#
#         # self.layers[-1] -= c*np.outer(output_deltas, np.append(hidden_outputs, 1))
#         # self.layers[0] -= c*np.outer(hidden_deltas, np.append(input_vector, 1))
#
#     def predict(self, input_vector):
#         deltas = [np.zeros([self.num_hidden]), np.zeros([self.output_size])]
#         return self.feed_forward(input_vector, deltas)[-1]
#
#     def predict_one(self, input_vector):
#         deltas = [np.zeros([self.num_hidden]), np.zeros([self.output_size])]
#         return np.argmax(self.feed_forward(input_vector, deltas)[-1])
#
#     def sauv(self, file_name=''):
#         if file_name == '':
#             file_name = 'nt_'+str(self.accu(TESTING)[0])
#         sauvfile = self.layers
#         f = open(file_name, 'wb')
#         pickle.dump(sauvfile, f)
#         f.close()
#
#     def load(self, file_name):
#         f = open(file_name, 'rb')
#         self.layers = pickle.load(f, encoding='latin1')
#         f.close()
#
#     def accu(self, testing):
#         """The lowest precision digit and total"""
#         res = np.zeros((10, 2))
#         for k in range(len(testing[1])):
#             if self.predict_one(testing[0][k]) == testing[1][k]:
#                 res[testing[1][k]] += 1
#             else:
#                 res[testing[1][k]][1] += 1
#         total = np.sum(res, axis=0)
#         each = [res[k][0]/res[k][1] for k in range(len(res))]
#         min_c = sorted(range(len(each)), key=lambda k: each[k])[0]
#         return np.round([each[min_c]*100, total[0]/total[1]*100, min_c], 2)
#
# nt1=Network(500)
# # nt1.train(60000,TRAINING, 5)
#
# per = nt1.train(60000,TRAINING)
#
# np.save('emstdp1_acc.npy', per)
#
# # =====================
# #   Display fonctions
# # =====================
#
# # Rounding off the prints and scientific notation
# np.set_printoptions(precision=2)
# np.set_printoptions(suppress=True)
#
#
# def find(c, network):
#     x = randint(0, 999)
#     while TESTING[1][x] != c:
#         x = randint(0, 10000)
#     aff(x, network)
#
#
# def aff(x, network):
#     print('Display character #'+str(x))
#     print('Target = '+str(TESTING[1][x]))
#     char = TESTING[0][x]
#     l = ''
#     for i in range(784):
#         if i % 28 == 0:
#             print(l)
#             l = str(int(round(char[i])))
#         else:
#             l += str(int(round(char[i])))
#     pred = network.predict(char)
#     print('Prediction = ' + str(np.argmax(pred)))
#     print(pred)
#
#
# def err(network):
#     x = randint(0, 10000)
#     while network.predict_one(TESTING[0][x]) == TESTING[1][x]:
#         x = randint(0, 10000)
#     aff(x, network)
#
#
# def test_nn(network):
#     """Test Network"""
#     ok, nb = 0, 10000
#     for k in range(nb):
#         if network.predict_one(TESTING[0][k]) == TESTING[1][k]:
#             ok += 1
#     return round((ok*100./nb), 1)
#
# # =====================
# #     Try with png
# # =====================
#
# def load_png(png):
#     img = misc.imread(png)
#     res = np.zeros(28*28)
#     for i, _ in enumerate(img):
#         for j, px in enumerate(img[i]):
#             res[28*i + j] = str(int(round(abs(px[1]-255)/255.)))
#     return res
#
#
# def aff2(x, *network):
#     char = x
#     l = ''
#     for i in range(784):
#         if i % 28 == 0:
#             print(l)
#             l = str(int(round(char[i])))
#         else:
#             l += str(int(round(char[i])))
#     for nt in network:
#         pred = nt.predict(char)
#         print('Prediction = ' + str(np.argmax(pred)))
#         print(pred)
