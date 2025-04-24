import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union

class LSTMModel(nn.Module):
    """
    Recurrent neural network for modeling temporal dependencies in asset returns.
    
    This model captures time-based patterns in returns and is useful for
    rolling predictions and signal memory. It represents the fifth level of
    sophistication in the model progression.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dim: int,
                num_layers: int = 2,
                output_dim: int = 1,
                dropout_rate: float = 0.3,
                bidirectional: bool = False,
                use_attention: bool = False):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state in LSTM
            num_layers: Number of LSTM layers
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism over LSTM outputs
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        if use_attention:
            self.attention_dim = hidden_dim * (2 if bidirectional else 1)
            self.attention = nn.Sequential(
                nn.Linear(self.attention_dim, self.attention_dim // 2),
                nn.Tanh(),
                nn.Linear(self.attention_dim // 2, 1)
            )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if len(x.shape) == 2:  # (batch_size, input_dim)
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        
        if self.use_attention:
            attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
            context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_dim * num_directions)
            out = context_vector
        else:
            out = lstm_out[:, -1, :]  # (batch_size, hidden_dim * num_directions)
        
        out = self.dropout(out)
        
        out = self.fc(out)  # (batch_size, output_dim)
        
        return out
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability (if attention is used).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Attention weights tensor of shape (batch_size, seq_len, 1)
        """
        if not self.use_attention:
            raise ValueError("Attention mechanism is not enabled for this model")
        
        if len(x.shape) == 2:  # (batch_size, input_dim)
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
            
            attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        return attention_weights
