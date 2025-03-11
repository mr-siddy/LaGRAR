# models/encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GCNEncoder(nn.Module):
    """GCN-based encoder for variational graph representation learning.
    
    This encoder maps node features to latent space parameters (mu, log_var).
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        """Initialize GCN encoder.
        
        Args:
            in_channels: Dimension of input node features
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of latent space
            dropout: Dropout probability
        """
        super(GCNEncoder, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass through the encoder.
        
        Args:
            x: Node features of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            mu: Mean vectors of shape [num_nodes, out_channels]
            logvar: Log variance vectors of shape [num_nodes, out_channels]
        """
        x = self.gcn1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        mu = self.gcn_mu(x, edge_index, edge_weight)
        logvar = self.gcn_logvar(x, edge_index, edge_weight)
        
        return mu, logvar

class GATEncoder(nn.Module):
    """GAT-based encoder for variational graph representation learning.
    
    This encoder maps node features to latent space parameters (mu, log_var).
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        """Initialize GAT encoder.
        
        Args:
            in_channels: Dimension of input node features
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of latent space
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.gat_mu = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        self.gat_logvar = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass through the encoder.
        
        Args:
            x: Node features of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            mu: Mean vectors of shape [num_nodes, out_channels]
            logvar: Log variance vectors of shape [num_nodes, out_channels]
        """
        # Edge weights not directly used in GAT (attention computed instead)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        mu = self.gat_mu(x, edge_index)
        logvar = self.gat_logvar(x, edge_index)
        
        return mu, logvar