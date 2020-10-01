import torch.nn as nn
import torch.nn.functional as F
from util.utils import *
import argparse

class Model(nn.Module):

    def __init__(self, params):

        super(Model, self).__init__()
        self.window = params.window
        self.horizon = params.horizon
        self.m = 1
        self.hidC = params.hidCNN
        self.hidR = params.hidRNN
        self.Ck = params.CNN_kernel
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.linear1 = nn.Linear(self.hidR, self.horizon)
        self.dropout = nn.Dropout(p=params.dropout)

        if params.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        if params.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        # CNN
        c = x.view(-1, 1, self.window, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()

        _, r = self.GRU1(r)

        r = self.dropout(torch.squeeze(r, 0))

        res = self.linear1(r)

        if self.output:
            res = self.output(res)

        res = torch.unsqueeze(res, 0)
        res = res.permute(1, 2, 0)

        return res


def create_model():
    parser = argparse.ArgumentParser(description='pytocrh time series forecasting')
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--hidCNN', type=int, default=100)
    parser.add_argument('--hidRNN', type=int, default=100)
    parser.add_argument('--window', type=int, default=36)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--CNN_kernel', type=int, default=6)
    parser.add_argument('--ephocs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--save', type=str, default='model/model.pt')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--normalize', type=int, default=1)
    parser.add_argument('--output_fun', type=str, default='sigmoid')

    params = parser.parse_args()

    params.cuda = params.gpu is not None
    if params.cuda:
        torch.cuda.set_device(params.gpu)

    Data = DataUtility(params)

    print('-----Model-------')

    model = Model(params)







