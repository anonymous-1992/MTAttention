import torch.utils.data
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class transDataset(Dataset):
    '''
    returns [samples, labels]
    '''

    def __init__(self, max_samples, input_size, output_size, time_steps, num_encoder_steps, data):

        self.input_size = input_size
        self.output_size = output_size
        self.time_steps = time_steps
        self.max_samples = max_samples
        self.data = data
        self.num_encoder_steps = num_encoder_steps

        len_data = len(data)
        dim = int(len_data / (self.input_size + self.output_size))
        X = torch.zeros((dim, self.input_size))
        Y = torch.zeros((dim, self.output_size))

        ind = 0
        for i in range(0, len_data, input_size + output_size):
            start = i
            end = start + self.input_size
            if end <= len_data:

                X[ind, :] = torch.from_numpy(data[start:end])
                Y[ind, :] = torch.from_numpy(data[end:end + self.output_size])
            ind += 1

        num_samples = min(self.max_samples, len_data)

        self.inputs = torch.zeros((num_samples, self.time_steps, self.input_size))
        self.outputs = torch.zeros((num_samples, self.time_steps, self.output_size))

        for i in range(num_samples):
            start = i
            end = start + self.time_steps
            if len(X[start:end, :]) == self.time_steps:
                self.inputs[i, :, :] = X[start:end, :]
                self.outputs[i, :, :] = Y[start:end, :]


    def __getitem__(self, index):
        s = {
            'inputs': self.inputs[index],
            'outputs': self.outputs[index],
        }
        return s

    def __len__(self):
        return self.inputs.shape[0]
