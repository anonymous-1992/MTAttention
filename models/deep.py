import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.linear1 = nn.Linear(self.hidR, self.m)
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

        return res





