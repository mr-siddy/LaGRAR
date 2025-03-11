# models/base.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GCN(nn.Module):
    """Graph Convolutional Network base model.
    
    Standard GCN implementation for comparison with LaGRAR.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=2):
        """Initialize GCN model.
        
        Args:
            in_channels: Dimension of input node features
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of output features
            dropout: Dropout probability
            num_layers: Number of GCN layers
        """
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass through the GCN.
        
        Args:
            x: Node features of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            Node embeddings after GCN layers
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x
    
    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Get node embeddings from the penultimate layer.
        
        Args:
            x: Node features of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            Node embeddings from the penultimate layer
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            if i < len(self.convs) - 2:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class GAT(nn.Module):
    """Graph Attention Network base model.
    
    Standard GAT implementation for comparison with LaGRAR.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5, num_layers=2):
        """Initialize GAT model.
        
        Args:
            in_channels: Dimension of input node features
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of output features
            heads: Number of attention heads
            dropout: Dropout probability
            num_layers: Number of GAT layers
        """
        super(GAT, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass through the GAT.
        
        Args:
            x: Node features of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            Node embeddings after GAT layers
        """
        # Edge weights not directly used in GAT (attention computed instead)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Get node embeddings from the penultimate layer.
        
        Args:
            x: Node features of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            Node embeddings from the penultimate layer
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            if i < len(self.convs) - 2:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x