import torch
import torch.nn as nn
from typing import Tuple, List
from torch.nn.utils.parametrizations import weight_norm

class MLPClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(MLPClassifier, self).__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # last time step
        return self.fc(out)

__all__ = ["MLPClassifier", "LSTMClassifier"]