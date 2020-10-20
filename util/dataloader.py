import torch.utils.data
import torch
from torch.utils.data import Dataset


class Data_Utility(Dataset):
    '''
    returns [samples, labels]
    '''

    def __init__(self, input_size, output_size, encode_length, time_steps, data):

        self.input_size = input_size
        self.output_size = output_size
        self.time_steps = time_steps
        self.encode_length = encode_length

        window = self.time_steps
        horizon = self.time_steps - self.encode_length

        x_shape = len(data) - window - horizon
        X = torch.zeros((x_shape, window))
        Y = torch.zeros((x_shape, horizon))

        for i in range(x_shape):
            start = i
            end = start + window
            X[i, :] = torch.from_numpy(data[start:end])
            Y[i, :] = torch.from_numpy(data[end:end + horizon])

        self.inputs = torch.reshape(X, (x_shape, window, self.input_size))
        self.outputs = torch.reshape(Y, (x_shape, horizon, self.output_size))

    def __getitem__(self, index):
        s = {
            'inputs': self.inputs[index],
            'outputs': self.outputs[index],
        }
        return s

    def __len__(self):
        return self.inputs.shape[0]
