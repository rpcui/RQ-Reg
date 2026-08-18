import torch
import torch.nn as nn

class LSTMModule(nn.Module):
    def __init__(self, config):
        super(LSTMModule, self).__init__()

        self.lstm = nn.LSTM(
            input_size=config['d_model'],
            hidden_size=config['d_model'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config['dropout']
        )

    def forward(self, x, hx):
        output, (h_n, c_n) = self.lstm(x, hx)
        return output, (h_n, c_n)
