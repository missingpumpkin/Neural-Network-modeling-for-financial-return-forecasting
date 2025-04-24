import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union

class DeepMLPModel(nn.Module):
    """
    Deeper MLP with dropout and batch normalization layers for better regularization.
    
    This model improves upon the basic MLP by adding more layers, batch normalization,
    and dropout for better regularization. It represents the third level of sophistication
    in the model progression, testing the impact of depth on performance.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int],
                output_dim: int = 1,
                dropout_rate: float = 0.3,
                use_batch_norm: bool = True,
                activation: str = 'relu'):
        """
        Initialize the Deep MLP model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions (deeper than basic MLP)
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function to use ('relu', 'leaky_relu', 'elu')
        """
        super(DeepMLPModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        layers = []
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self._get_activation(activation))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(self._get_activation(activation))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function based on name."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if len(x.shape) == 3:  # (batch_size, seq_len, input_dim)
            x = x[:, -1, :]  # Shape: (batch_size, input_dim)
        
        return self.network(x)
