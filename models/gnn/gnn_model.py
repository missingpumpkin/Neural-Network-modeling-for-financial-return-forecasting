import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union

class GNNLayer(nn.Module):
    """
    Graph Neural Network layer for asset pricing.
    """
    def __init__(self, in_features: int, out_features: int):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN layer.
        
        Args:
            x: Node features tensor of shape (batch_size, num_nodes, in_features)
            adj: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
            
        Returns:
            Updated node features of shape (batch_size, num_nodes, out_features)
        """
        support = torch.matmul(x, self.weight)  # (batch_size, num_nodes, out_features)
        
        output = torch.matmul(adj, support)  # (batch_size, num_nodes, out_features)
        
        output = output + self.bias
        
        return output

class GNNModel(nn.Module):
    """
    Graph neural network over asset similarity graphs (e.g., sectors, correlations).
    
    This model captures interdependencies between assets and enables factor propagation
    and sector-aware learning. It represents the sixth level of sophistication in the
    model progression.
    """
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int],
                output_dim: int = 1,
                dropout_rate: float = 0.3,
                graph_type: str = 'correlation',
                num_gnn_layers: int = 2,
                pooling: str = 'mean'):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (typically 1 for return prediction)
            dropout_rate: Dropout rate for regularization
            graph_type: Type of graph to construct ('correlation', 'sector', 'knn')
            num_gnn_layers: Number of GNN layers
            pooling: Pooling method for graph-level output ('mean', 'max', 'sum')
        """
        super(GNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.graph_type = graph_type
        self.num_gnn_layers = num_gnn_layers
        self.pooling = pooling
        
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = hidden_dims[0] if i == 0 else hidden_dims[min(i, len(hidden_dims) - 1)]
            out_dim = hidden_dims[min(i + 1, len(hidden_dims) - 1)]
            self.gnn_layers.append(GNNLayer(in_dim, out_dim))
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dims[min(i + 1, len(hidden_dims) - 1)])
            for i in range(num_gnn_layers)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def construct_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Construct adjacency matrix based on the specified graph type.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Adjacency matrix of shape (batch_size, seq_len, seq_len) or (batch_size, 1, 1)
        """
        if len(x.shape) == 2:  # (batch_size, input_dim)
            batch_size = x.shape[0]
            return torch.eye(1).unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
        
        batch_size, seq_len, _ = x.shape
        
        if self.graph_type == 'correlation':
            x_centered = x - x.mean(dim=2, keepdim=True)
            x_normalized = x_centered / (x_centered.norm(dim=2, keepdim=True) + 1e-8)
            adj = torch.matmul(x_normalized, x_normalized.transpose(1, 2))
            
            adj = F.softmax(adj, dim=2)
            
        elif self.graph_type == 'knn':
            x_flat = x.view(batch_size, seq_len, -1)
            x_square = torch.sum(x_flat ** 2, dim=2, keepdim=True)
            x_square_t = x_square.transpose(1, 2)
            x_dot = torch.matmul(x_flat, x_flat.transpose(1, 2))
            dist = x_square + x_square_t - 2 * x_dot
            
            k = min(5, seq_len - 1)
            _, indices = torch.topk(-dist, k=k, dim=2)
            adj = torch.zeros(batch_size, seq_len, seq_len).to(x.device)
            for b in range(batch_size):
                for i in range(seq_len):
                    adj[b, i, indices[b, i]] = 1.0
            
            adj = (adj + adj.transpose(1, 2)) / 2
            
            adj = adj + torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
            
            rowsum = adj.sum(dim=2, keepdim=True) + 1e-8
            adj = adj / rowsum
            
        else:  # Default to identity (no graph structure)
            adj = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
        
        return adj
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if len(x.shape) == 2:  # (batch_size, input_dim)
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        adj = self.construct_graph(x)  # (batch_size, seq_len, seq_len)
        
        h = self.input_proj(x)  # (batch_size, seq_len, hidden_dims[0])
        h = F.relu(h)
        h = self.dropout(h)
        
        for i, (gnn_layer, batch_norm) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            h_new = gnn_layer(h, adj)  # (batch_size, seq_len, hidden_dims[i+1])
            
            h_new_trans = h_new.transpose(1, 2)  # (batch_size, hidden_dims[i+1], seq_len)
            h_new_norm = batch_norm(h_new_trans)
            h_new = h_new_norm.transpose(1, 2)  # (batch_size, seq_len, hidden_dims[i+1])
            
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            if h.shape[-1] == h_new.shape[-1]:
                h = h + h_new
            else:
                h = h_new
        
        if self.pooling == 'mean':
            h_graph = torch.mean(h, dim=1)  # (batch_size, hidden_dims[-1])
        elif self.pooling == 'max':
            h_graph, _ = torch.max(h, dim=1)  # (batch_size, hidden_dims[-1])
        elif self.pooling == 'sum':
            h_graph = torch.sum(h, dim=1)  # (batch_size, hidden_dims[-1])
        else:
            h_graph = h[:, -1, :]  # (batch_size, hidden_dims[-1])
        
        out = self.mlp(h_graph)  # (batch_size, output_dim)
        
        return out
