import pandas as pd
import torch
from torch.autograd import Variable
import os
import numpy as np


class TransDataset(object):

    def __init__(self,  max_samples,
                 input_size, output_size, cuda):
        self.input_size = input_size
        self.output_size = output_size
        self.cuda = cuda
        self.max_samples = max_samples

    def get_batches(self, data, i):

        X, Y = list(), list()
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + self.input_size
            out_end = in_end + self.output_size
            if out_end <= len(data):
                X.append(data[in_start:in_end])
                Y.append(data[in_end:out_end])
            in_start += 1
        X = np.array(X)
        Y = np.array(Y)
        in_size = min(self.max_samples, len(data))
        inputs = torch.zeros((in_size, self.batch_size, self.input_size))
        outputs = torch.zeros((in_size, self.batch_size, self.output_size))

        inputs[i, :, :] = torch.from_numpy(X[i:i + self.batch_size, :])
        outputs[i, :, :] = torch.from_numpy(Y[i:i + self.batch_size, :])

        yield inputs, outputs

    '''def get_batches(self, inputs, target, batch_size, shuffle=True):

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
            start_idx += batch_size'''

