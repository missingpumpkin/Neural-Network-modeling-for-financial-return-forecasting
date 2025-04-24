import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union

class DeepFactorNetwork(nn.Module):
    """
    Deep neural network for asset pricing based on multiple factors.
    
    This model implements a deep learning approach to asset pricing,
    allowing for complex non-linear relationships between factors and returns.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int] = [64, 32],
                output_dim: int = 1,
                dropout_rate: float = 0.2,
                use_batch_norm: bool = True):
        """
        Initialize the deep factor network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(DeepFactorNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        layers = []
        
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            
            layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if len(x.shape) == 3:  # (batch_size, seq_len, input_dim)
            # For BatchNorm1d, we need (batch_size, input_dim)
            x = x[:, -1, :]  # Take the last time step
        
        return self.model(x)


class LSTMFactorNetwork(nn.Module):
    """
    LSTM-based neural network for asset pricing with time series factors.
    
    This model uses LSTM layers to capture temporal dependencies in financial data,
    combined with factor information for asset pricing.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dim: int = 64,
                num_layers: int = 2,
                output_dim: int = 1,
                dropout_rate: float = 0.2,
                bidirectional: bool = False):
        """
        Initialize the LSTM factor network.
        
        Args:
            input_dim: Dimension of input features at each time step
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMFactorNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if len(x.shape) == 2:  # (batch_size, input_dim)
            x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, input_dim)
        
        lstm_out, _ = self.lstm(x)
        
        lstm_out = lstm_out[:, -1, :]
        
        output = self.fc(lstm_out)
        
        return output


class TemporalFactorNetwork(nn.Module):
    """
    Temporal convolutional network for asset pricing with time series factors.
    
    This model uses 1D convolutions to capture temporal patterns in financial data,
    combined with factor information for asset pricing.
    """
    
    def __init__(self, 
                input_dim: int,
                seq_len: int,
                num_filters: List[int] = [64, 128, 64],
                kernel_sizes: List[int] = [3, 3, 3],
                output_dim: int = 1,
                dropout_rate: float = 0.2):
        """
        Initialize the temporal factor network.
        
        Args:
            input_dim: Dimension of input features at each time step
            seq_len: Length of input sequence (time steps)
            num_filters: List of filter counts for each convolutional layer
            kernel_sizes: List of kernel sizes for each convolutional layer
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
        """
        super(TemporalFactorNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        conv_layers = []
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                conv_layers.append(nn.Dropout(dropout_rate))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.flat_size = num_filters[-1] * seq_len
        
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = x.permute(0, 2, 1)
        
        x = self.conv_layers(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class HybridFactorNetwork(nn.Module):
    """
    Hybrid neural network combining LSTM and feedforward layers for asset pricing.
    
    This model uses LSTM layers to process temporal data and feedforward layers
    to incorporate static factors, providing a comprehensive approach to asset pricing.
    """
    
    def __init__(self, 
                temporal_input_dim: int,
                static_input_dim: int,
                hidden_dim: int = 64,
                num_layers: int = 2,
                fc_dims: List[int] = [32, 16],
                output_dim: int = 1,
                dropout_rate: float = 0.2):
        """
        Initialize the hybrid factor network.
        
        Args:
            temporal_input_dim: Dimension of temporal input features at each time step
            static_input_dim: Dimension of static input features
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            fc_dims: Dimensions of fully connected layers after LSTM
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
        """
        super(HybridFactorNetwork, self).__init__()
        
        self.temporal_input_dim = temporal_input_dim
        self.static_input_dim = static_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.fc_dims = fc_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self.lstm = nn.LSTM(
            input_size=temporal_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        fc_layers = []
        prev_dim = hidden_dim + static_input_dim  # Combine LSTM output with static features
        
        for dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = dim
        
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, temporal_x: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            temporal_x: Temporal input tensor of shape (batch_size, seq_len, temporal_input_dim)
            static_x: Static input tensor of shape (batch_size, static_input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        lstm_out, _ = self.lstm(temporal_x)
        
        lstm_out = lstm_out[:, -1, :]
        
        combined = torch.cat([lstm_out, static_x], dim=1)
        
        output = self.fc_layers(combined)
        
        return output
