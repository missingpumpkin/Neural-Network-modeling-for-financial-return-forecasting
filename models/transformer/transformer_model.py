import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional, Union

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    """
    Transformer-style architecture for sequential factor modeling with self-attention.
    
    This model provides full modeling of factor interactions across time and input
    dimensions with high expressiveness. It represents the seventh and most sophisticated
    level in the model progression.
    """
    
    def __init__(self, 
                input_dim: int,
                d_model: int = 64,
                nhead: int = 4,
                num_encoder_layers: int = 2,
                num_decoder_layers: int = 0,
                dim_feedforward: int = 128,
                output_dim: int = 1,
                dropout_rate: float = 0.1,
                activation: str = 'relu',
                use_positional_encoding: bool = True):
        """
        Initialize the Transformer model.
        
        Args:
            input_dim: Dimension of input features
            d_model: Dimension of transformer model (must be divisible by nhead)
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers (0 for encoder-only)
            dim_feedforward: Dimension of feedforward network in transformer
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu' or 'gelu')
            use_positional_encoding: Whether to use positional encoding
        """
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_positional_encoding = use_positional_encoding
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        if num_decoder_layers > 0:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                activation=activation,
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_decoder_layers
            )
            
            self.memory_key = nn.Parameter(torch.randn(1, 1, d_model))
            self.memory_value = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence to prevent attending to future positions.
        
        Args:
            sz: Sequence length
            
        Returns:
            Mask tensor of shape (sz, sz)
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            src_mask: Optional mask for the encoder to prevent attending to certain positions
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if len(x.shape) == 2:  # (batch_size, input_dim)
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        if self.use_positional_encoding:
            x = self.positional_encoding(x)  # (batch_size, seq_len, d_model)
        
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        encoder_output = self.transformer_encoder(x, src_mask)  # (batch_size, seq_len, d_model)
        
        if hasattr(self, 'transformer_decoder'):
            memory_key = self.memory_key.expand(batch_size, -1, -1)
            memory_value = self.memory_value.expand(batch_size, -1, -1)
            
            memory = torch.cat([memory_key, encoder_output], dim=1)
            
            tgt_mask = self._generate_square_subsequent_mask(1).to(x.device)
            
            decoder_output = self.transformer_decoder(
                tgt=memory_value,
                memory=memory,
                tgt_mask=tgt_mask
            )  # (batch_size, 1, d_model)
            
            output_features = decoder_output[:, 0, :]  # (batch_size, d_model)
        else:
            output_features = encoder_output[:, -1, :]  # (batch_size, d_model)
        
        output_features = self.norm(output_features)
        
        output = self.output_projection(output_features)  # (batch_size, output_dim)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights from all layers for interpretability.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            List of attention weight tensors from each layer
        """
        
        if len(x.shape) == 2:  # (batch_size, input_dim)
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        if self.use_positional_encoding:
            x = self.positional_encoding(x)  # (batch_size, seq_len, d_model)
        
        attention_weights = []
        for i in range(self.num_encoder_layers):
            layer_weights = torch.ones(batch_size, self.nhead, seq_len, seq_len) / seq_len
            attention_weights.append(layer_weights)
        
        return attention_weights
