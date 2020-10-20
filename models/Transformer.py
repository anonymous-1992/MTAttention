import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
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
                 , d_forward, window, horizon, input_size, output_size,
                 encode_length, dropout, add_ar):
        super(Trns_Model, self).__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.output_size = output_size
        self.encode_length = encode_length
        self.window = window
        self.horizon = horizon
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
