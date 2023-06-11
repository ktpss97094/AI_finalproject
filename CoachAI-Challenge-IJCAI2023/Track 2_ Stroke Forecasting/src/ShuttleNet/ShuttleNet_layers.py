import torch.nn as nn
import torch
from ShuttleNet.ShuttleNet_submodules import TypeAreaMultiHeadAttention, MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.disentangled_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        '''
        Convolutional Unit
        '''
        self.dim3_Conv_F1_1 = nn.Conv1d(3, 64, 3, dilation=1, padding='same').to('cuda')
        self.dim3_Conv_F1_2 = nn.Conv1d(3, 64, 3, dilation=1, padding='same').to('cuda')
        self.dim2_Conv_F1_1 = nn.Conv1d(2, 64, 3, dilation=1, padding='same').to('cuda')
        self.dim2_Conv_F1_2 = nn.Conv1d(2, 64, 3, dilation=1, padding='same').to('cuda')
        self.dim1_Conv_F1_1 = nn.Conv1d(1, 64, 3, dilation=1, padding='same').to('cuda')
        self.dim1_Conv_F1_2 = nn.Conv1d(1, 64, 3, dilation=1, padding='same').to('cuda')
        self.BatchNorm_F1 = nn.BatchNorm1d(64).to('cuda')
        self.Conv_F2_1 = nn.Conv1d(64, 32, 3, dilation=2, padding='same').to('cuda')
        self.Conv_F2_2 = nn.Conv1d(64, 32, 3, dilation=2, padding='same').to('cuda')
        self.BatchNorm_F2 = nn.BatchNorm1d(32).to('cuda')
        self.Conv_F3_1 = nn.Conv1d(32, 16, 3, dilation=3, padding='same').to('cuda')
        self.Conv_F3_2 = nn.Conv1d(32, 16, 3, dilation=3, padding='same').to('cuda')
        self.BatchNorm_F3 = nn.BatchNorm1d(16).to('cuda')

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, encode_area, encode_shot, slf_attn_mask=None):
        encode_output, enc_slf_attn, enc_disentangled_weight = self.disentangled_attention(encode_area, encode_area, encode_area, encode_shot, encode_shot, encode_shot, mask=slf_attn_mask)  # (32, 2, 32)
        
        left_1, right_1 = None, None
        if encode_output.shape[1] == 3:
            left_1 = nn.Tanh()(self.dim3_Conv_F1_1(encode_output))
            right_1 = nn.Sigmoid()(self.dim3_Conv_F1_2(encode_output))
        elif encode_output.shape[1] == 2:
            left_1 = nn.Tanh()(self.dim2_Conv_F1_1(encode_output))
            right_1 = nn.Sigmoid()(self.dim2_Conv_F1_2(encode_output))
        elif encode_output.shape[1] == 1:
            left_1 = nn.Tanh()(self.dim1_Conv_F1_1(encode_output))
            right_1 = nn.Sigmoid()(self.dim1_Conv_F1_2(encode_output))
        else:
            raise NotImplementedError('encode_output.shape[1]非3或2或1')
        F1 = self.BatchNorm_F1(torch.mul(left_1, right_1))

        left_2 = nn.Tanh()(self.Conv_F2_1(F1))
        right_2 = nn.Sigmoid()(self.Conv_F2_2(F1))
        F2 = self.BatchNorm_F2(torch.mul(left_2, right_2))

        left_3 = nn.Tanh()(self.Conv_F3_1(F2))
        right_3 = nn.Sigmoid()(self.Conv_F3_2(F2))
        F3 = self.BatchNorm_F3(torch.mul(left_3, right_3))

        encode_output = torch.cat((F1, F2, F3, encode_output), dim=1)

        encode_output = self.pos_ffn(encode_output)  # (32, 2, 32)
        return encode_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.decoder_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, decode_area, decode_shot, encode_output, slf_attn_mask=None, dec_enc_attn_mask=None, return_attns=None):
        decode_output, dec_slf_attn, disentangled_weight = self.decoder_attention(decode_area, decode_area, decode_area, decode_shot, decode_shot, decode_shot, mask=slf_attn_mask, return_attns=return_attns)
        decode_output, dec_enc_slf_attn = self.decoder_encoder_attention(decode_output, encode_output, encode_output, mask=dec_enc_attn_mask)
        decode_output = self.pos_ffn(decode_output)
        return decode_output, dec_slf_attn, dec_enc_slf_attn, disentangled_weight


class GatedFusionLayer(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner*n, d, bias=False)
        self.gated2 = nn.Linear(d_inner*n, d, bias=False)
        self.gated3 = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A, x_B, x_L):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A.shape
        w_A = self.w_A.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B = self.w_B.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L = self.w_L.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A = self.tanh_f(self.hidden1(x_A))
        h_B = self.tanh_f(self.hidden2(x_B))
        h_L = self.tanh_f(self.hidden3(x_L))

        x = torch.cat((x_A, x_B, x_L), dim=-1)
        z1 = self.sigmoid_f(self.gated1(x)) * h_A
        z2 = self.sigmoid_f(self.gated2(x)) * h_B
        z3 = self.sigmoid_f(self.gated3(x)) * h_L

        z1 = w_A[:, :seq_len, :] * z1
        z2 = w_B[:, :seq_len, :] * z2
        z3 = w_L[:, :seq_len, :] * z3

        return self.sigmoid_f(z1 + z2 + z3)