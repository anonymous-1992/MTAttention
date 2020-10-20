import torch.nn as nn
import torch
from models.Transformer import PositionalEncoding, AR
from torch.nn import LayerNorm
import torch.nn.functional as F


class LSTM_enc_dec(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout,
                 time_steps, encode_length, input_size, output_size):
        super(LSTM_enc_dec, self).__init__()
        self.lstm_encoder = nn.LSTM(input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout=dropout)
        self.lstm_decoder = nn.LSTM(input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout=dropout)

        self.lstm_layers = num_layers
        self.hidden_size = hidden_size
        self.position_encode = PositionalEncoding(hidden_size, dropout)
        self.norm = LayerNorm(hidden_size)
        self.encode_length = encode_length
        self.enc_input_embed = nn.Linear(input_size, self.hidden_size)
        self.dec_input_embed = nn.Linear(input_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, output_size)
        self.ar = AR(time_steps, time_steps - encode_length)

    def encode(self, x, batch_size, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.lstm_layers, batch_size, self.hidden_size)

        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))

        return output, hidden

    def decode(self, x, batch_size, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.lstm_layers, batch_size, self.hidden_size)

        output, (hidden, cell) = self.lstm_decoder(x, (hidden, hidden))

        return output, hidden

    def forward(self, src):
        batch_size = src.size(0)
        enc_input = src[:, :self.encode_length, :].permute(1, 0, 2)
        dec_input = src[:, self.encode_length:, :].permute(1, 0, 2)

        # shift outputs to the right by 1
        dec_input = torch.roll(dec_input, shifts=(0, 0, 1), dims=(0, 1, 2))

        enc_input = self.enc_input_embed(enc_input)
        dec_input = self.dec_input_embed(dec_input)

        enc_input = self.position_encode(enc_input)
        dec_input = self.position_encode(dec_input)

        enc_output, hidden = self.encode(enc_input, batch_size)
        dec_output, _ = self.decode(dec_input, batch_size, hidden)
        dec_output = dec_output.permute(1, 0, 2)
        dec_output = self.norm(dec_output)
        dec_output = self.linear_out(dec_output)
        ar_output = self.ar(src)
        final_output = dec_output + ar_output

        return final_output


class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, input_size, output_size, input_seq_len, output_seq_len):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.linear_2 = nn.Linear(input_seq_len, output_seq_len)
        self.output = F.sigmoid
        self.lstm_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):

        x = x.permute(1, 0, 2)
        res, _ = self.lstm(x)
        res = self.linear_1(res)
        res = res.permute(1, 0, 2)
        res = self.linear_2(res.permute(0, 2, 1))
        res = res.permute(0, 2, 1)
        return res


class LSTNet(nn.Module):

    def __init__(self, params):

        super(LSTNet, self).__init__()
        self.window = params.time_steps
        self.horizon = params.time_steps - params.encode_length
        self.m = 1
        self.hidC = params.hidCNN
        self.hidR = params.hidRNN
        self.Ck = params.CNN_kernel
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.linear1 = nn.Linear(self.hidR, self.horizon)
        self.dropout = nn.Dropout(p=params.dropout_rate)

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