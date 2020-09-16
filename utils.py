import torch
import numpy as np
import pandas as pd
import os
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


class DataUtility(object):

    def __init__(self, params):
        self.cuda = params.cuda
        self.window = params.window
        self.horizon = params.horizon
        self.normalize = params.normalize
        self.data_dir = params.data_dir
        self.train = self.get_samples('train')
        self.validation = self.get_samples('validation')
        self.test = self.get_samples('test')
        self.n_std = normal_std(self.test[1])
        self.scale = 1

    def get_samples(self, type):

        data_path = os.path.join(self.data_dir, '{}.csv'.format(type))
        data = pd.read_csv(data_path)
        data = np.array(data['SpConductivity'].values)
        data = data.reshape(data.shape[0], 1)
        if self.normalize == 0:
            data = data
            self.scale = 1.
        if self.normalize == 1:
            data = data / np.max(data)
            self.scale = np.max(data)
        batches = self._batcher(data)
        return batches

    def _batcher(self, data_set):

        n = len(data_set)
        X = torch.zeros((n, self.window, 1))
        Y = torch.zeros((n, 1))

        for i in range(n):
            start = i
            end = i + self.window
            if end+self.horizon <= n:
                X[i, :, :] = torch.from_numpy(data_set[start:end, :])
                Y[i, :] = torch.from_numpy(data_set[end+self.horizon - 1])
        return [X, Y]

    def get_batches(self, inputs, target, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))

        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = target[excerpt]
            if self.cuda:
                X = X.cuda()
                Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size
