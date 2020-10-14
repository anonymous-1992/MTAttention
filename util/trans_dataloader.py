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

    def __init__(self, max_samples, window, horizon, input_size, output_size, time_steps, num_encoder_steps, data):

        self.window = window
        self.horizon = horizon
        self.input_size = input_size
        self.output_size = output_size
        self.time_steps = time_steps
        self.max_samples = max_samples
        self.data = data.reshape(data.shape[0], 1)
        self.num_encoder_steps = num_encoder_steps

        len_data = len(data)
        self.inputs = torch.zeros((len_data, self.window, self.input_size))
        self.outputs = torch.zeros((len_data, self.horizon, self.output_size))

        for i in range(len_data):
            start = i
            end = i + self.input_size
            if end+self.output_size <= len_data:
                self.inputs[i, :, :] = torch.from_numpy(self.data[start:end, :])
                self.outputs[i, :, :] = torch.from_numpy(self.data[end:end + self.output_size, :])
        '''X = torch.zeros((len_data, self.input_size))
        Y = torch.zeros((len_data, self.output_size))

        for i in range(len_data):
            start = i
            end = start + self.input_size
            if end + self.output_size <= len_data:
                X[i, :] = torch.from_numpy(data[start:end])
                Y[i, :] = torch.from_numpy(data[end:end + self.output_size])'''

        '''num_samples = int(len_data / self.time_steps)

        self.inputs = torch.zeros((num_samples, self.time_steps, self.input_size))
        self.outputs = torch.zeros((num_samples, self.time_steps, self.output_size))

        for i in range(num_samples):
            start = i
            end = start + self.time_steps
            if len(X[start:end, :]) == self.time_steps:
                self.inputs[i, :, :] = X[start:end, :]
                self.outputs[i, :, :] = Y[start:end, :]'''

    def __getitem__(self, index):
        s = {
            'inputs': self.inputs[index],
            'outputs': self.outputs[index],
        }
        return s

    def __len__(self):
        return self.inputs.shape[0]
