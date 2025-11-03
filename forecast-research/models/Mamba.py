import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from layers.Embed import DataEmbedding

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16) # TODO implement "auto"

        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )

        # Prediction head: project from last hidden state to future predictions
        # This enables forecasting any horizon length (not limited by seq_len)
        self.prediction_head = nn.Linear(configs.d_model, configs.pred_len * configs.c_out)

        

    def forecast(self, x_enc, x_mark_enc):
        # Normalize input
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Embed and process with Mamba
        x = self.embedding(x_enc, x_mark_enc)  # [batch, seq_len, d_model]
        x = self.mamba(x)  # [batch, seq_len, d_model]

        # Take the last hidden state (contains info about entire input sequence)
        # State-space models like Mamba capture long-range dependencies in final state
        last_hidden = x[:, -1, :]  # [batch, d_model]

        # Project to future predictions
        x_out = self.prediction_head(last_hidden)  # [batch, pred_len * c_out]
        x_out = x_out.reshape(-1, self.pred_len, self.c_out)  # [batch, pred_len, c_out]

        # Denormalize
        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            # Forecast already returns exactly pred_len future timesteps
            x_out = self.forecast(x_enc, x_mark_enc)  # [batch, pred_len, c_out]
            return x_out

        # other tasks not implemented
