import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union

class MLPModel(nn.Module):
    """
    Simple feedforward neural network with 2-3 layers and ReLU activation.
    
    This model introduces nonlinearity to capture factor-return relationships
    beyond linear regression, representing the second level of sophistication
    in the model progression.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int],
                output_dim: int = 1,
                dropout_rate: float = 0.2,
                use_batch_norm: bool = True):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(MLPModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        layers = []
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
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
