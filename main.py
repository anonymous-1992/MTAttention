import argparse
import time
from torch.nn import functional as F
from models.deep import Model


from utils import *
from torch import optim

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


def train(data, X, Y, model, optimizer, batch_size):

    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        output = model(X)
        loss = F.mse_loss(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_samples += 1
    return total_loss / n_samples


def evaluate(data, X, Y, model, batch_size):
    model.eval()
    total_loss = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        total_loss += F.mse_loss(output * data.scale, Y * data.scale)
        n_samples += 1

    loss_mean = total_loss / n_samples
    predict = predict.view(-1, 1)
    test = test.view(-1, 1)

    y_diff = predict - test
    y_mean = torch.mean(test)
    y_trans = test - y_mean

    val_rrse = torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_trans, 2)))

    y_m = torch.mean(test, 0, True)
    y_hat_m = torch.mean(predict, 0, True)
    y_d = test - y_m
    y_hat_d = predict - y_hat_m
    corr_num = torch.sum(y_d * y_hat_d, 0)
    corr_denom = torch.sqrt((torch.sum(torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)))
    corr_inter = corr_num / corr_denom
    val_corr = torch.sum(corr_inter)

    return loss_mean, val_rrse, val_corr


def main():
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
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--save', type=str, default='model/model.pt')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--normalize', type=int, default=1)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    params = parser.parse_args()

    params.cuda = params.gpu is not None
    if params.cuda:
        torch.cuda.set_device('cuda')

    Data = DataUtility(params)

    print('-----Model-------')
    model = Model(params)

    optimizer = optim.Adam(model.parameters())

    best_val = float("inf")

    print('---Training----')
    for ephoc in range(1, params.ephocs + 1):

        ephoc_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, optimizer, params.batch_size)
        val_rmse, val_rse, val_corr = evaluate(Data, Data.validation[0], Data.validation[1], model, params.batch_size)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} |  '
              'rmse {:5.4f} |  rse {:5.4f} |  corr {:5.4f}'.
              format(ephoc, (time.time() - ephoc_start_time), train_loss, val_rmse, val_rse, val_corr))

        if val_rmse < best_val:
            with open(params.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_rmse

    with open(params.save, 'rb') as f:
        model = torch.load(f)
    test_rmse, test_rse, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, params.batch_size)
    print('|  Test rmse {:5.4f} |  Test rse {:5.4f} |  Test corr {:5.4f}'.format(test_rmse, test_rse, test_corr))


if __name__ == '__main__':
    main()