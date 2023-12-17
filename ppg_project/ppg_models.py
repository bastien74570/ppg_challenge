"""Module defining PPG models."""
import numpy as np
import torch
from torch import nn
from typing import Tuple
import numpy as np


class SpectralCNN(nn.Module):
    """Spectral CNN model definition."""

    def __init__(
        self,
        input_data: Tuple,
        hidden_dim: int = 1,
        n_units: int = 8,
        kernel_size: int = 3,
        hidden_size: int = 64,
        batch_norm: bool = True,
        max_pool: bool = True,
        dropout: bool = True,
        
    ):
        """Initialize instance of SpectralCNN.

        Args:
            input_data (Tuple): input data shape.
            hidden_dim (int): number of hidden layers. Defaults to 1.
            n_units (int): number of units for conv layers. Defaults to 8.
            kernel_size (int): kernel size to use. Defaults to 3.
            hidden_size (int): number of units for final fc layer. Defaults to 64.
            dilation (int): dilation to use. Defaults to 1.
            batch_norm (bool): whether to use batch normalization. Defaults to True.
            max_pool (bool): whether to use max pooling. Defaults to True.
            dropout (bool): whether to use dropout. Defaults to True.
        """
        super().__init__()

        self.conv_layers = nn.ModuleList()

        in_channels = 1
        out_channels = n_units

        h_out, w_out = input_data[0], input_data[1]

        for i in range(hidden_dim):
            list_layers_conv = []
            list_layers_conv.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
            )
            h_out = h_out - kernel_size + 1
            w_out = w_out - kernel_size + 1

            if batch_norm:
                list_layers_conv.append(nn.BatchNorm2d(out_channels))
            list_layers_conv.append(nn.ReLU())

            if max_pool:
                list_layers_conv.append(nn.MaxPool2d(2))
                h_out = np.floor(h_out / 2)
                w_out = np.floor(w_out / 2)

            if dropout:
                list_layers_conv.append(nn.Dropout(0.2))
            conv = nn.Sequential(*list_layers_conv)
            in_channels = out_channels
            out_channels = out_channels * 2
            self.conv_layers.append(conv)

        self.fc1 = nn.Sequential(
            nn.Linear(int(h_out * w_out * in_channels), hidden_size)
        )
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply model forward pass.

        Args:
            x (torch.Tensor): input data as tensor.

        Returns:
            torch.Tensor: tensor of predictions.
        """
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
