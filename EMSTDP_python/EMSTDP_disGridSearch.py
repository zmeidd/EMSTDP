
import os
# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["MKL_DOMAIN_NUM_THREADS"] = "MKL_BLAS=8"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_DYNAMIC"] = "False"
# os.environ["OMP_DYNAMIC"] = "False"

import sys
sys.path.append('./')

import numpy as np
# import BPSNN_allSpikeErr_batchstdp2 as svp
import EMSTDP_algo as svp
import matplotlib.pyplot as plt
import cPickle as pickle
import gzip
import time
import random

from tqdm import tqdm, tqdm_gui, tnrange, tgrange, trange
# import progressbar

import argparse

import ray
from ray.tune import grid_search, run_experiments
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining
# ray.init()
from tensorboard_logger import Logger, configure, log_value

num_cpus = "1"

total_train_size = 60000
total_test_size = 10000
ver_period = 60000

train_size = 10000
test_size = 10000
ver_period = train_size

data_index = (np.linspace(0, total_train_size - 1, total_train_size)).astype(int)

# h = [300, 300]#[100,300,500,700,test_size,1500]

def combine_matrix(*args):
    n=len(args)
    rows,cols=args[0].shape
    nn = (np.ceil(np.sqrt(n))).astype(int)
    a=np.zeros((nn*rows, nn*cols))
    m=0
    for i in range(nn):
        for j in range(nn):
            a[i*rows:(i+1)*rows,j*cols:(j+1)*cols]=args[m]
            m+=1
            if m>=n:
                return a
    return a

def scale(x, out_range=(0, 1), axis=None):
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

class SNN_model(Trainable):
    def _read_data(self):
        # START_TIME = time.time()
        from keras.datasets import mnist  # load MNIST dataset
        # from keras.datasets import fashion_mnist as mnist     # load Fashion MNIST dataset
        
        all_data = mnist.load_data()
        TRAINING = all_data[0]
        TESTING = all_data[1]

        # data_index = (np.linspace(0, train_size - 1, train_size)).astype(int)
        x_train = np.expand_dims(np.reshape(TRAINING[0][0:total_train_size], [total_train_size, 784]),
                              axis=0).astype(float) / 255.0
        y_train = np.zeros((total_train_size, 10))
        for i in range(total_train_size):
            y_train[i, TRAINING[1][i]] = 1
        y_train = np.argmax(y_train, axis=1).astype(int)

        x_test = np.expand_dims(np.reshape(TESTING[0][0:total_test_size], [total_test_size, 784]),
                                  axis=0).astype(float) / 255.0
        y_test = np.zeros((total_test_size, 10))
        for i in range(total_test_size):
            y_test[i, TESTING[1][i]] = 1
        y_test = np.argmax(y_test, axis=1).astype(int)

        return (x_train, y_train), (x_test, y_test)

    def _setup(self):
        import os
        os.environ["MKL_NUM_THREADS"] = num_cpus
        os.environ["MKL_DOMAIN_NUM_THREADS"] = "MKL_BLAS="+num_cpus
        os.environ["OMP_NUM_THREADS"] = num_cpus
        os.environ["MKL_DYNAMIC"] = "False"
        os.environ["OMP_DYNAMIC"] = "False"

        self.train_data, self.test_data = self._read_data()

        self.epoch = 0
        self.flip = 1.0
        # [500], [400, 300], [350, 300, 200], [300,250,250,200]

        if self.config["hl"] == 1:
            self.h = [500]
        if self.config["hl"] == 2:
            self.h = [500, 500]
        if self.config["hl"] == 3:
            self.h = [500, 500, 500]
        if self.config["hl"] == 4:
            self.h = [300,300,300,300]

        self.model = svp.Network(self.config["dfa"],self.config["dropr"],self.config["evt"], self.config["norm"], self.config["rel"], self.config["delt"], self.config["dr"], self.config["init"], self.config["clp"], self.config["lim"], 784, self.h, 10, self.config["hThr"] * self.config["fr"], self.config["outputThr"] * self.config["fr"])


    def _train(self):
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data

        self.epoch += 1

        s_index = data_index

        self.model.T = self.config["T"]

        self.model.bias = self.config["bias"]
        self.model.lr = self.config["lr"]

        fr = self.config["fr"]

        batch_size = self.config["batch_size"]

        self.model.twin = self.model.T / 2 - self.config["twin"]
        self.model.epsilon = self.config["epsilon"]
        self.model.scale = self.config["scale"]

        tmp_d = np.zeros([self.model.T, batch_size, 784])

        test_acc = []
        train_acc = []
        err = np.zeros([train_size])

        tbs = 1000


        for ep in range(1):
            np.random.shuffle(s_index)
            pred1 = np.zeros([train_size])

            spikes = np.zeros([self.model.T, batch_size, 784]).astype(float)
            spikes2 = np.zeros([self.model.T, batch_size, 784]).astype(float)
            for i in trange(train_size/batch_size):
                tmp_rand = np.random.random([self.model.T, 1, 1])
                randy = np.tile(tmp_rand, (1, batch_size, 784))
                tmp_d = np.tile(x_train[:, s_index[i * batch_size:(i + 1) * batch_size], :], (self.model.T, 1, 1))
                spikes = randy < (tmp_d * fr)

                pred1[i*batch_size:(i+1)*batch_size],err[i*batch_size:(i+1)*batch_size] = self.model.Train(spikes.astype(float),(y_train[s_index[i* batch_size:(i + 1) * batch_size]]), batch_size)

                # sys.stdout.flush()


                if ((i+1)*batch_size % ver_period == 0):  # 5000
                    pred = np.zeros([test_size])
                    for i2 in range(test_size/tbs):  # train_size
                        tmp_rand = np.random.random([self.model.T, 1, 1])
                        randy = np.tile(tmp_rand, (1, tbs, 784))
                        tmp_d = np.tile(x_test[:, i2 * tbs:(i2 + 1) * tbs, :], (self.model.T, 1, 1))
                        spikes2 = randy < (tmp_d * fr)
                        pred[i2 * tbs:(i2 + 1) * tbs] = self.model.Test(spikes2.astype(float),tbs)
                    acn = sum(pred == y_test[0:test_size]) / float(test_size)
                    print str(self.epoch) + " test_accuray " + str(acn) + " LR " + str(self.model.lr)
                    test_acc.append(sum(pred == y_test[0:test_size]) / float(test_size))

            train_acc.append(sum(pred1 == y_train[s_index[0:train_size]]) / float(train_size))
            error = np.mean(err)
            print(" LR " + str(self.model.lr))

        return {"train_accuracy": np.mean(train_acc), "test_accuracy":np.mean(test_acc), "training_error":error}

    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + '/ray_SNN_W_epoch_' + str(self.epoch) + '.pickle'
        # self.model.save_weights(file_path)
        if self.epoch%20 == 0:
            with open(file_path , 'w') as fw:
                save = {'w_h': self.model.w_h,
                        'w_o': self.model.w_o}
                pickle.dump(save, fw)

        # writer = SummaryWriter(checkpoint_dir)

        logger = Logger(checkpoint_dir+'/images/')
        # logger_hist = Logger(checkpoint_dir + '/histograms/')
        # img = np.random.rand(10, 10)
        images = []
        # mx = np.max(self.w_h[0])
        # mn = np.min(self.w_h[0])
        for tp in range(self.model.outputs):
            tpp = self.model.w_o[:, tp]
            sz = np.ceil((np.sqrt(np.size(tpp)))).astype(int)
            tpm = np.zeros(sz * sz, dtype=float)
            tpm[0:len(tpp)] = tpp
            tpp = tpm
            tpp = np.reshape(tpp, newshape=[sz, sz])
            images.append(scale(tpp))
        cimages = combine_matrix(*images)
        logger.log_images('key_out', [cimages], step=self.epoch)
        logger.log_histogram('key_out', [self.model.w_o], step=self.epoch)


        for k in range(len(self.h)):
            images = []
            for tp in range(self.h[k]):
                tpp = self.model.w_h[k][:, tp]
                sz = np.ceil((np.sqrt(np.size(tpp)))).astype(int)
                tpm = np.zeros(sz*sz, dtype=float)
                tpm[0:np.size(tpp)] = tpp
                tpp = tpm
                tpp = np.reshape(tpp, newshape=[sz, sz])
                images.append(scale(tpp))
            cimages = combine_matrix(*images)
            logger.log_images('key_'+str(k), [cimages], step=self.epoch)
            logger.log_histogram('key'+str(k), [self.model.w_h[k]], step=self.epoch)


        return file_path


    def _restore(self, path):
        load_weight = True
        load_epoch = 1
        if load_weight == True:
            with open(path, 'rb') as f2:
                save2 = pickle.load(f2)
                self.model.w_h = save2['w_h']
                self.model.w_o = save2['w_o']


    def _stop(self):
        # If need, save your model when exit.
        # saved_path = self.model.save(self.logdir)
        # print("save model at: ", saved_path)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()

    train_spec = {
        "run": SNN_model,
        "trial_resources": {
            "cpu": 1
        },
        "stop": {
            "test_accuracy": 0.99,
            "training_iteration": 3000,
        },
        "config": {
            "T": 200,
            "dfa": 0,
            "hl": grid_search([1, 2]),
            "dropr": 0.0,
            "evt": 0,
            "lim": 1.0,
            "norm": 0.0,
            "rel": 0,
            "delt": 5,
            "dr": 1,
            "init": 0,
            "clp": 1,
            "hThr": grid_search([0.2, 0.4]),
            "outputThr": 1.0,
            "batch_size": 10,
            "fr": 1.0,
            "bias": 0.0,
            "twin": 1,
            "scale": 1,
            "epsilon": grid_search([1, 3, 5]),
            "lr": 0.001
        },
        "num_samples": 1, "checkpoint_freq": 1, "checkpoint_at_end": True, "max_failures": 2,
    }

    # if args.smoke_test:
    #     train_spec["config"]["lr"] = 10**-5
        # train_spec["config"]["dropout"] = 0.5

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="test_accuracy",
        perturbation_interval=1000,
        hyperparam_mutations={
            "lr": [1e-5, 1e-6]
        }
    )

    run_experiments({"SNN_model_test": train_spec}, scheduler=pbt)
