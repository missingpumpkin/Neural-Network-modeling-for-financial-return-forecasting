import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union

class LinearModel(nn.Module):
    """
    Linear model for asset pricing based on multiple factors.
    
    This model implements a linear approach to asset pricing,
    similar to traditional Fama-French-style linear asset pricing models.
    It has no hidden layers and serves as a baseline for comparison.
    """
    
    def __init__(self, 
                input_dim: int,
                output_dim: int = 1):
        """
        Initialize the linear factor model.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (typically 1 for return prediction)
        """
        super(LinearModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
    
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
        
        return self.linear(x)
