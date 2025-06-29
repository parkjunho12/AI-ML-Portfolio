import torch
import torch.nn as nn
from typing import Tuple, List
from torch.nn.utils.parametrizations import weight_norm
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import utils
import snntorch.functional as SF
import snntorch.spikegen as spikegen
from snntorch import surrogate

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

class Chomp1d(nn.Module):
    """
    Remove extra padding from the right side
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """
    Temporal Block for TCN implementation
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, 
                 dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, 
                                                   dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.spike1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, 
                                                   dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.spike2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        mem1 = self.spike1.init_leaky()
        mem2 = self.spike2.init_leaky()

        x1 = self.conv1(x)
        x1 = self.chomp1(x1)
        mem1, spk1 = self.spike1(x1, mem1)
        x1 = self.relu1(spk1)
        x1 = self.dropout1(x1)

        x2 = self.conv2(x1)
        x2 = self.chomp2(x2)
        mem2, spk2 = self.spike2(x2, mem2)
        x2 = self.relu1(spk2)
        out = self.dropout2(x2)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for NinaPro dataset
    """
    def __init__(self, num_inputs: int, num_channels: List[int], num_classes: int, 
                 kernel_size: int = 2, dropout: float = 0.2):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        # TCN expects: (batch_size, num_features, seq_len)
        x = x.unsqueeze(2) 
        
        # Pass through TCN
        y = self.network(x)
        
        # Global average pooling
        y = torch.mean(y, dim=2)  # (batch_size, num_channels[-1])
        
        # Classification
        return self.classifier(y)


__all__ = ["Chomp1d", "TemporalBlock", "TCN"]