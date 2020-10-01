import argparse
from torch import Tensor
import torch.nn as nn
from util.trans_dataloader import TransDataset
from torch.distributions import TransformedDistribution
import torch
import math
from main import run_models


class TransformerNetwork(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 act_type: str,
                 dropout_rate: float,
                 dim_feedforward_scale: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 input_size: int,
                 encode_length: int
                 ) -> None:
        super().__init__()

        self.encode_length = encode_length
        self.d_model = d_model

        # [B, T, d_model] where d_model / num_heads is int
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward_scale * d_model,
            dropout=dropout_rate,
            activation=act_type
        )

        # mask
        self.register_buffer(
            "tgt_mask", self.transformer.generate_square_subsequent_mask(input_size)
        )

    def forward(self, inputs) -> Tensor:

        '''
        :param input: (batch_size, len, dim) shape
        :return:
        '''

        enc_input = inputs[:, :self.encode_length, ...].permute(0, 2, 1)

        dec_input = inputs[:, self.encode_length:, ...].permute(0, 2, 1)

        encoder_input = nn.Linear(enc_input.size(2), self.d_model)

        # pass through encoder
        enc_out = self.transformer.encoder(
            encoder_input(enc_input).permute(1, 0, 2)
        )

        decoder_input = nn.Linear(dec_input.size(2), self.d_model)

        # input to decoder
        dec_output = self.transformer.decoder(
            decoder_input(dec_input).permute(1, 0, 2),
            enc_out,  # memory
            tgt_mask=self.tgt_mask,
        )

        return dec_output


def main():

    parser = argparse.ArgumentParser(description='pytocrh time series forecasting Transformers')
    parser.add_argument('--data_dir', type=str, default="../data")
    parser.add_argument('--input_size', type=int, default=5, help='window')
    parser.add_argument('--output_size', type=int, default=1, help='horizon')
    parser.add_argument('--time_steps', type=int, default=124)
    parser.add_argument('--encode_length', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--act_type', type=str, default='relu')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--dim_feedforward_scale', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--ephocs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    params = parser.parse_args()

    d_model = params.d_model
    num_heads = params.num_heads
    act_type = params.act_type
    dropout_rate = params.dropout_rate
    dim_feedforward_scale = params.dim_feedforward_scale
    num_encoder_layers = params.num_encoder_layers
    num_decoder_layers = params.num_decoder_layers
    input_size = params.input_size
    encode_length = params.encode_length

    Data = TransDataset(params.time_steps, params.max_samples, params.input_size, params.output_size, params.data_dir, False)

    transformer = TransformerNetwork(
        d_model,
        num_heads,
        act_type,
        dropout_rate,
        dim_feedforward_scale,
        num_encoder_layers,
        num_decoder_layers,
        input_size,
        encode_length,

    )

    run_models(transformer, Data, params)


if __name__ == '__main__':
    main()















