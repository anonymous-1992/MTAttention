import argparse
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from util import trans_dataloader
import matplotlib.pyplot as plt
from torch.nn import LayerNorm, Dropout


class PositionwiseFeedForward(nn.Module):

    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_forward, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.pos_fnn = PositionwiseFeedForward(d_model, d_forward)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(self, enc_input):

        enc_output, self_attn_weights = self.self_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_fnn(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        return enc_output


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_forward, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.pos_fnn = PositionwiseFeedForward(d_model, d_forward)
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)

    def forward(self, dec_input, enc_output):

        dec_output, slf_attn_weights = self.slf_attn(dec_input, dec_input, dec_input)
        dec_output = self.dropout_1(dec_output)
        dec_output = self.norm_1(dec_output)
        dec_output, multi_attn_weights = self.multihead_attn(dec_output, enc_output, enc_output)
        dec_output = self.pos_fnn(dec_output)
        dec_output = self.dropout_2(dec_output)
        dec_output = self.norm_2(dec_output)

        return dec_output


class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.layer_stacks = nn.ModuleList([
            encoder_layer for _ in range(num_layers)
        ])

    def forward(self, input):

        output = input
        for enc_layer in self.layer_stacks:
            output = enc_layer(output)

        return output


class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):

        super(Decoder, self).__init__()
        self.layer_stacks = nn.ModuleList([
            decoder_layer for _ in range(num_layers)
        ])

    def forward(self, target, memory):

        output = target

        for dec_layer in self.layer_stacks:
            output = dec_layer(output, memory)

        return output


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


class AR(nn.Module):

    def __init__(self, window, horizon):

        super(AR, self).__init__()
        self.linear = nn.Linear(window, horizon)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = x.permute(0, 2, 1)
        return x


class Trns_Model(nn.Module):

    def __init__(self, d_model, n_head, num_enc_layers, num_dec_layers
                 , d_forward, window, horizon, input_size, output_size, encode_length, dropout, add_ar):
        super(Trns_Model, self).__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.output_size = output_size
        self.encode_length = encode_length
        self.add_ar = add_ar
        self.position_encoding = PositionalEncoding(d_model, dropout)
        self.enc_input_embed = nn.Linear(self.input_size, self.d_model)
        self.dec_input_embed = nn.Linear(self.input_size, self.d_model)
        self.ar = AR(window, horizon)
        self.linear_out = nn.Linear(d_model, output_size)
        enc_layer = EncoderLayer(d_model, d_forward, n_head, dropout)
        self.encoder = Encoder(enc_layer, num_enc_layers)
        dec_layer = DecoderLayer(d_model, d_forward, n_head, dropout)
        self.decoder = Decoder(dec_layer, num_dec_layers)

    def forward(self, src):

        enc_input = src[:, :self.encode_length, :].permute(1, 0, 2)
        dec_input = src[:, self.encode_length:, :].permute(1, 0, 2)

        # shift outputs to the right by 1
        dec_input = torch.roll(dec_input, shifts=(0, 0, 1), dims=(0, 1, 2))

        enc_input = self.enc_input_embed(enc_input)
        dec_input = self.dec_input_embed(dec_input)

        enc_input = self.position_encoding(enc_input)
        dec_input = self.position_encoding(dec_input)

        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_output)

        output = dec_output.permute(1, 0, 2)
        decoder_output = self.linear_out(output)

        if self.add_ar:
            ar_output = self.ar(src)
            final_output = decoder_output + ar_output
        else:
            final_output = decoder_output

        return final_output


def get_configs():

    parser = argparse.ArgumentParser(description='pytocrh time series forecasting Transformers')
    parser.add_argument('--data_dir', type=str, default="../data/split_ds/")
    parser.add_argument('--site', type=str, default="BDCs_1")
    parser.add_argument('--input_size', type=int, default=1, help='n_features')
    parser.add_argument('--output_size', type=int, default=1, help='target shape')
    parser.add_argument('--window', type=int, default=128, help='window')
    parser.add_argument('--horizon', type=int, default=28, help='horizon')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--act_type', type=str, default='relu')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--d_forward', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--ephocs', type=int, default=50)
    parser.add_argument('--time_steps', type=int, default=128)
    parser.add_argument('--encode_length', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--prediction_length', type=int, default=128)
    parser.add_argument('--save', type=str, default="Model")
    parser.add_argument('--add_ar', type=bool, default=True)
    parser.add_argument('--trns_fname', type=str, default="trns_ar")
    parser.add_argument('--ar_fname', type=str, default="ar")
    params = parser.parse_args()
    return params


def data_loader(params, set_type):

    data_path = os.path.join(params.data_dir, '{}.csv'.format(params.site))
    data = pd.read_csv(data_path)
    data = np.array(data['SpConductivity'].values)
    data = data[:-2 * params.prediction_length] if set_type == 'train' else \
        data[-2 * params.prediction_length:- params.prediction_length] if set_type == 'valid' else \
        data[-params.prediction_length:]
    dataset = trans_dataloader.transDataset(params.max_samples, params.window, params.horizon,
              params.input_size, params.output_size, params.time_steps,
                           params.encode_length, data)

    loader = DataLoader(
        dataset=dataset,
        batch_size=params.batch_size
    )

    return loader


def create_plot(labels, predictions, rmse, pred_len, fname):

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, pred_len), labels.detach().numpy(), color='lime', label="Measurements")
    ax.plot(np.arange(0, pred_len), predictions.detach().numpy(), color='darkorange',
            label="Predictions rmse: %.2f" % rmse)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s.svg" % fname)


def evaluate(params, model, eval_loader):

    model.eval()
    total_samples = 0
    total_loss = 0
    predict = None
    labels = None
    criterion = nn.MSELoss()
    for i, batch in enumerate(eval_loader):
        output = model(batch['inputs'])
        if predict is None:
            predict = output
            labels = batch['outputs']
        else:
            predict = torch.cat((predict, output))
            labels = torch.cat((labels, batch['outputs']))

        total_loss += criterion(predict, labels)
        total_samples += params.batch_size

    predict = predict.reshape((len(predict) * params.horizon, ))
    labels = labels.reshape((len(labels) * params.horizon, ))
    rmse = math.sqrt(total_loss / total_samples)

    y_diff = predict - labels
    y_mean = torch.mean(labels)
    y_trans = labels - y_mean

    rrse = torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_trans, 2)))

    y_m = torch.mean(labels, 0, True)
    y_m_hat = torch.mean(predict, 0, True)
    y_d = labels - y_m
    y_d_hat = predict - y_m_hat
    corr_num = torch.sum(y_d * y_d_hat, 0)
    corr_denom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_d_hat, 2), 0)))
    corr_inter = corr_num / corr_denom
    corr = torch.sum(corr_inter)

    return rmse, rrse, corr, predict, labels


def train(params, model, train_loader, criterion):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_samples = 0
    total_loss = 0
    samples = 0
    start = time.time()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch['inputs'])
        loss = criterion(output, batch['outputs'])
        loss.backward()
        optimizer.step()
        total_loss += loss
        total_samples += params.batch_size
        samples += params.batch_size
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / params.batch_size, samples / elapsed))
            start = time.time()
            samples = 0
    return total_loss / total_samples


def run_models(params, model, fname):

    train_iter = data_loader(params, 'train')
    valid_iter = data_loader(params, 'valid')
    test_iter = data_loader(params, 'test')

    criterion = nn.MSELoss()
    best_val = float("inf")
    for i in range(params.ephocs):
        model.train()
        loss = train(params, model, train_iter, criterion)
        print('train loss: {:5.2f}'.format(loss))

        rmse, rrse, corr, _, _ = evaluate(params, model, valid_iter)
        print('validation rmse: {:5.2f}, validation rrse: {:5.2f}, validation corr: {:5.2f}'.
              format(rmse, rrse, corr))

        if rmse < best_val:
            with open(params.save, 'wb') as f:
                torch.save(model, f)
            best_val = rmse

    with open(params.save, 'rb') as f:
        model = torch.load(f)
    rmse, rrse, corr, predict, labels = evaluate(params, model, test_iter)
    print('test mse: {:5.2f}, test rrse: {:5.2f}, test corr: {:5.2f}'.format(rmse, rrse, corr))
    create_plot(labels, predict, rmse, len(predict) * params.horizon, fname)


def main():

    params = get_configs()
    trans_model = Trns_Model(params.d_model, params.n_head, params.num_encoder_layers,
                  params.num_decoder_layers, params.d_forward, params.window, params.horizon, params.input_size,
                  params.output_size, params.encode_length, params.dropout_rate, params.add_ar)

    run_models(params, trans_model, params.trns_fname)

    ar_model = AR(params.window, params.horizon)

    run_models(params, ar_model, params.ar_fname)


if __name__ == '__main__':
    main()













