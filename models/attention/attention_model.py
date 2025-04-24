import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union

class AttentionNet(nn.Module):
    """
    Feature-attentive MLP that learns importance weights over input factors or time steps.
    
    This model adds interpretability and dynamic factor weighting to the deep MLP,
    representing the fourth level of sophistication in the model progression.
    It matches recent deep learning finance research by incorporating attention
    mechanisms to identify the most relevant factors for prediction.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int],
                output_dim: int = 1,
                dropout_rate: float = 0.3,
                use_batch_norm: bool = True,
                attention_dim: int = 32,
                num_heads: int = 1):
        """
        Initialize the Attention Network model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            attention_dim: Dimension of attention mechanism
            num_heads: Number of attention heads (1 for single-head attention)
        """
        super(AttentionNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        if num_heads == 1:
            self.query = nn.Linear(input_dim, attention_dim)
            self.key = nn.Linear(input_dim, attention_dim)
            self.value = nn.Linear(input_dim, input_dim)
        else:
            self.mha = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
        
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
    
    def attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism to input features.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        if self.num_heads == 1:
            if len(x.shape) == 2:  # (batch_size, input_dim)
                x_reshaped = x.unsqueeze(1)
            else:  # (batch_size, seq_len, input_dim)
                x_reshaped = x
                
            batch_size, seq_len, _ = x_reshaped.shape
            
            q = self.query(x_reshaped)  # (batch_size, seq_len, attention_dim)
            k = self.key(x_reshaped)    # (batch_size, seq_len, attention_dim)
            v = self.value(x_reshaped)  # (batch_size, seq_len, input_dim)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attention_dim ** 0.5)  # (batch_size, seq_len, seq_len)
            
            attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
            
            attended_features = torch.matmul(attention_weights, v)  # (batch_size, seq_len, input_dim)
            
            if len(x.shape) == 2:
                return attended_features.squeeze(1), attention_weights.squeeze(1)
            else:
                return attended_features, attention_weights
        else:
            if len(x.shape) == 2:  # (batch_size, input_dim)
                x_reshaped = x.unsqueeze(1)
            else:  # (batch_size, seq_len, input_dim)
                x_reshaped = x
                
            attended_features, attention_weights = self.mha(
                query=x_reshaped,
                key=x_reshaped,
                value=x_reshaped,
                need_weights=True
            )
            
            if len(x.shape) == 2:
                return attended_features.squeeze(1), attention_weights.squeeze(1)
            else:
                return attended_features, attention_weights
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        attended_x, attention_weights = self.attention(x)
        
        if len(x.shape) == 3:  # (batch_size, seq_len, input_dim)
            attended_x = attended_x[:, -1, :]  # Shape: (batch_size, input_dim)
        
        output = self.network(attended_x)
        
        return output, attention_weights
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Attention weights tensor
        """
        with torch.no_grad():
            _, attention_weights = self.attention(x)
        
        return attention_weights
