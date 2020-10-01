import argparse
import math
import time
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import os
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from torch.nn import LayerNorm
import numpy as np


class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout, activation, encode_length, input_size, output_size):
        super(TransformerModel, self).__init__()
        self.encode_length = encode_length
        self.position_encoding = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        self.src_mask = None
        self.input_size = input_size
        self.output_size = output_size
        self.softmax = nn.Softmax()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.register_buffer(
            "tgt_mask", self._generate_square_subsequent_mask(decode_length)
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        '''if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask
        '''
        src = src.permute(1, 0, 2)

        src_encode = src[:self.encode_length, :, :]
        src_decode = src[self.encode_length:, :, :]

        # shift outputs to the right by 1
        src_decode = torch.roll(src_decode, shifts=(0, 0, -1), dims=(0, 1, 2))

        encoder = nn.Linear(self.input_size, self.d_model)
        decoder = nn.Linear(self.input_size, self.d_model)

        encoder_input_emb = encoder(src_encode)
        decoder_input_emb = decoder(src_decode)

        encoder_input_emb = self.position_encoding(encoder_input_emb)
        decoder_input_emb = self.position_encoding(decoder_input_emb)

        memory = self.encoder(encoder_input_emb)
        dec_output = self.decoder(decoder_input_emb, memory, tgt_mask=self.tgt_mask,)

        output = torch.cat([memory, dec_output], dim=0)
        output_layer = nn.Linear(self.d_model, self.output_size)
        f_output = output_layer(output)
        f_output = self.softmax(f_output)

        return f_output


def make_std_mask(tgt, pad):
    " Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def subsequent_mask(size):
    " Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             - (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def get_data(data):
    X, Y = list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + params.input_size
        out_end = in_end + params.output_size
        if out_end <= len(data):
            X.append(data[in_start:in_end])
            Y.append(data[in_end:out_end])
        in_start = out_end
    X = np.array(X)
    Y = np.array(Y)
    in_size = min(max_samples, len(data))
    inputs = torch.zeros((in_size, params.time_steps, params.input_size))
    outputs = torch.zeros((in_size, params.time_steps, params.output_size))
    for i in range(in_size):
        if i + params.time_steps < in_size:
            inputs[i, :, :] = torch.from_numpy(X[i:i + params.time_steps, :])
            outputs[i, :, :] = torch.from_numpy(Y[i:i + params.time_steps, :])

    return [inputs, outputs]


def get_batches(inputs, target, batch_size, shuffle=False):

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
        yield X, Y
    start_idx += batch_size


parser = argparse.ArgumentParser(description='pytocrh time series forecasting Transformers')
parser.add_argument('--data_dir', type=str, default="../data/split_ds/")
parser.add_argument('--site', type=str, default="BDCs_1")
parser.add_argument('--input_size', type=int, default=5, help='window')
parser.add_argument('--output_size', type=int, default=1, help='horizon')
parser.add_argument('--d_model', type=int, default=32)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--act_type', type=str, default='relu')
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--dim_feedforward', type=int, default=4)
parser.add_argument('--num_encoder_layers', type=int, default=3)
parser.add_argument('--num_decoder_layers', type=int, default=3)
parser.add_argument('--ephocs', type=int, default=10)
parser.add_argument('--time_steps', type=int, default=198)
parser.add_argument('--encode_length', type=int, default=168)
parser.add_argument('--batch_size', type=int, default=64)
params = parser.parse_args()

decode_length = 30
max_samples = 1000
prediction_length = 128

data = pd.read_csv(os.path.join(params.data_dir, '{}.csv'.format(params.site)))
data = data["SpConductivity"].values
train, valid, test = data[: -2 * prediction_length], data[- 2 * prediction_length: -prediction_length], data[-prediction_length:]

criterion = nn.MSELoss()
model = TransformerModel(params.d_model, params.num_heads, params.num_encoder_layers, params.num_decoder_layers
                         , params.dim_feedforward, params.dropout_rate, params.act_type, params.encode_length, params.input_size, params.output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

train_iter = get_data(train)
valid_iter = get_data(valid)
test_iter = get_data(test)

epochs = 100

for i in range(epochs):

    " Standard Training and Logging Function"
    model.train()
    start = time.time()
    total_samples = 0
    total_loss = 0
    tokens = 0
    for X, Y in get_batches(train_iter[0], train_iter[1], params.batch_size):

        optimizer.zero_grad()
        out = model.forward(X)
        loss = criterion(out.permute(1, 0, 2), Y)
        loss.backward()
        optimizer.step()
        total_loss += loss
        total_samples += params.batch_size
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / params.batch_size, tokens / elapsed))
            start = time.time()
            tokens = 0
    print(total_loss / total_samples)



