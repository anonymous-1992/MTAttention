import torch
import math
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from util import dataloader
import matplotlib.pyplot as plt
import argparse
import os
import time
import torch.nn as nn
from models.Transformer import Trns_Model, AR
from models.deep_models import LSTM_enc_dec, LSTM, LSTNet
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class Data:

    def __init__(self, data_dir, site, max_samples, valid_len, test_len):
        data_path = os.path.join(data_dir, '{}.csv'.format(site))
        data = pd.read_csv(data_path)
        data = np.array(data['SpConductivity'].values)
        #data = data[- max_samples:]
        n = len(data)
        data_to_fit = data.reshape(-1, 1)
        self.scaler = StandardScaler().fit(data_to_fit)
        norm_data = self.scaler.transform(data_to_fit)
        norm_data = norm_data.reshape(len(norm_data), )
        train_len = n - valid_len - test_len
        valid = n - valid_len
        train_set = range(0, train_len)
        valid_set = range(train_len, valid)
        test_set = range(valid, n)
        self.train = norm_data[train_set]
        self.valid = norm_data[valid_set]
        self.test = norm_data[test_set]

    def denormalize(self, data):
        return self.scaler.inverse_transform(data)


def get_configs(parser):

    parser.add_argument('--data_dir', type=str, default="data/split_ds/")
    parser.add_argument('--site', type=str, default="BDCs_1")
    parser.add_argument('--input_size', type=int, default=1, help='n_features')
    parser.add_argument('--output_size', type=int, default=1, help='target shape')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--act_type', type=str, default='gelu')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--d_forward', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--ephocs', type=int, default=100)
    parser.add_argument('--time_steps', type=int, default=16)
    parser.add_argument('--encode_length', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--prediction_length', type=int, default=1000)
    parser.add_argument('--valid_length', type=int, default=1000)
    parser.add_argument('--save', type=str, default="Model")
    parser.add_argument('--add_ar', type=bool, default=True)
    params = parser.parse_args()
    return params


def get_data_set(params, dataset):
    return dataloader.Data_Utility(params.input_size,
                                   params.output_size,
                                   params.encode_length,
                                   params.time_steps,
                                   dataset)


def data_loader(params, data):

    data_loader = DataLoader(
        dataset=data,
        batch_size=params.batch_size
    )

    return data_loader


def create_plot(labels, predictions, fname):

    rmse = math.sqrt(mean_squared_error(labels, predictions))

    pred_len = len(predictions)
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, pred_len), labels, color='lime', label="Measurements")
    ax.plot(np.arange(0, pred_len), predictions, color='darkorange',
            label="Predictions rmse: %.2f" % rmse)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s.svg" % fname)


def corr(predict, Ytest):

    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return correlation


def eval_metrics(predict, label, test_data):

    data_rse = test_data.std() * np.sqrt((len(test_data) - 1.) / len(test_data))
    data_rae = np.mean(np.abs(test_data - np.mean(test_data)))

    loss = np.square(np.subtract(label, predict)).mean()
    n_samples = len(label)
    loss_mean = loss / n_samples

    rse = math.sqrt(loss / n_samples) / data_rse
    rae = (loss / n_samples) / data_rae
    correlation = corr(predict, label)
    return loss_mean, rse, rae, correlation


def evaluate(params, data, model, eval_loader, data_obj):

    model.eval()
    total_samples = 0
    total_loss = 0
    predictions = None
    labels = None
    data_rse = data.std() * np.sqrt((len(data) - 1.) / len(data))
    data_rae = np.mean(np.abs(data - np.mean(data)))

    criterion = nn.MSELoss()
    for i, batch in enumerate(eval_loader):
        output = model(batch['inputs'])
        pred = output.view(output.size(0) * output.size(1), -1)
        pred = data_obj.denormalize(pred.detach().numpy())
        pred = torch.from_numpy(pred)
        pred = pred.view(output.size(0), output.size(1), -1)

        label = batch["outputs"].view(output.size(0) * output.size(1), -1)
        label = data_obj.denormalize(label.detach().numpy())
        label = torch.from_numpy(label)
        label = label.view(output.size(0), output.size(1), -1)

        if predictions is None:
            predictions = pred
            labels = label
        else:
            predictions = torch.cat((predictions, output))
            labels = torch.cat((labels, batch['outputs']))

        total_loss += criterion(predictions, labels)
        total_samples += params.batch_size

    mse = total_loss / total_samples
    rse = math.sqrt(total_loss / total_samples) / data_rse
    rae = (total_loss / total_samples) / data_rae

    predict = predictions.data.cpu().numpy()
    Ytest = labels.data.cpu().numpy()
    correlation = corr(predict, Ytest)

    return mse, rse, rae, correlation


def predict(data, time_steps, encode_length, model):

    predictions = None
    labels = None
    horizon = time_steps - encode_length
    input_data = np.concatenate((data.valid[-time_steps:], data.test))
    output_data = data.valid
    model_input = torch.zeros((1, time_steps, 1))
    for i in range(0, len(data.test) - horizon, horizon):

        model_input[:, :, :] = torch.from_numpy(input_data[i:i+time_steps]).view(1, time_steps, 1)
        output = model(model_input)

        output_model = torch.from_numpy(output_data[i:i+horizon]).view(1, output.size(1), 1)

        if predictions is None:
            predictions = output
            labels = output_model
        else:
            predictions = torch.cat((predictions, output))
            labels = torch.cat((labels, output_model))

    predict_flat = predictions.view(predictions.size(0) * predictions.size(1), )
    labels_flat = labels.view(labels.size(0) * labels.size(1), )

    predict_flat = predict_flat.data.cpu().numpy()
    labels_flat = labels_flat.data.cpu().numpy()

    predict_val = data.denormalize(predict_flat)
    labels_val = data.denormalize(labels_flat)

    predict = predictions.data.cpu().numpy()
    label = labels.data.cpu().numpy()

    mse, rse, rae, correlation = eval_metrics(predict, label, data.test)
    return mse, rse, rae, correlation, predict_val, labels_val


def train(params, model, train_loader, criterion, data_obj, optimizer):

    total_samples = 0
    total_loss = 0
    samples = 0
    start = time.time()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch['inputs'])
        pred = output.view(output.size(0) * output.size(1), -1)
        y = batch['outputs']
        y = y.view(y.size(0) * y.size(1), -1)
        pred = data_obj.denormalize(pred.detach().numpy())
        y = data_obj.denormalize(y.detach().numpy())
        pred = torch.from_numpy(pred)
        y = torch.from_numpy(y)
        pred = pred.view(output.size(0), output.size(1), -1)
        y = y.view(output.size(0), output.size(1), -1)

        loss = criterion(pred, y)
        loss = Variable(loss, requires_grad=True)
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


def run_models(params, data, model, fname):

    train_set = get_data_set(params, data.train)
    valid_set = get_data_set(params, data.valid)

    train_iter = data_loader(params, train_set)
    valid_iter = data_loader(params, valid_set)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val = float("inf")
    for i in range(params.ephocs):
        model.train()
        loss = train(params, model, train_iter, criterion, data, optimizer)
        print('train loss: {:5.2f}'.format(loss))

        mse, rse, rae, corr = evaluate(params, data.valid, model, valid_iter, data_obj=data)
        print('validation mse: {:5.2f}, validation rse: {:5.2f}, validation rae: '
              '{:5.2f}, validation corr: {:5.2f}'.
              format(mse, rse, rae, corr))

        if mse < best_val:
            with open(params.save, 'wb') as f:
                torch.save(model, f)
            best_val = mse

    with open(params.save, 'rb') as f:
        model = torch.load(f)
    mse, rse, rae, cor, predictions, labels = \
        predict(data, params.time_steps, params.encode_length, model)

    create_plot(labels, predictions, fname)


def main():
    parser = argparse.ArgumentParser(description='pytocrh time series forecasting Transformers')
    params = get_configs(parser)
    data = Data(params.data_dir, params.site, params.max_samples,
                params.valid_length, params.prediction_length)
    trans_model = Trns_Model(
        params.d_model, params.n_head, params.num_encoder_layers,
        params.num_decoder_layers, params.d_forward, params.time_steps,
        params.time_steps - params.encode_length, params.input_size,
        params.output_size,
        params.encode_length, params.dropout_rate, params.add_ar)

    run_models(params, data, trans_model, "Transformer_AR")

    ar_model = AR(params.time_steps, params.time_steps - params.encode_length)

    run_models(params, data, ar_model, "AR")

    lstm_parser = parser
    lstm_parser.add_argument("--hidden_size", type=int, default=160)
    lstm_parser.add_argument("--num_lstm_layers", type=int, default=1)
    lstm_args = lstm_parser.parse_args()

    lstm_enc_dec_model = LSTM_enc_dec(lstm_args.hidden_size, lstm_args.num_lstm_layers,
                      lstm_args.dropout_rate,
                      lstm_args.time_steps,
                      lstm_args.encode_length,
                      lstm_args.input_size,
                      lstm_args.output_size)

    run_models(lstm_args, data, lstm_enc_dec_model, "LSTM_enc_dec")

    lstm_model = LSTM(lstm_args.hidden_size,
                      lstm_args.num_lstm_layers,
                      lstm_args.input_size,
                      lstm_args.output_size,
                      lstm_args.time_steps,
                      lstm_args.time_steps - lstm_args.encode_length)

    run_models(params, data, lstm_model, "LSTM")

    lstnet_parser = parser
    lstnet_parser.add_argument('--hidCNN', type=int, default=100)
    lstnet_parser.add_argument('--hidRNN', type=int, default=100)
    parser.add_argument('--CNN_kernel', type=int, default=6)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    lstnet_args = lstm_parser.parse_args()
    lstnet_model = LSTNet(lstnet_args)

    run_models(lstm_args, data, lstnet_model, "LSTNet")


if __name__ == '__main__':
    main()
