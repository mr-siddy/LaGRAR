# models/decoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.locality_preserving import LocalityPreservingSparsification

class InnerProductDecoder(nn.Module):
    """Inner product decoder for link prediction.
    
    This decoder reconstructs adjacency matrix from node embeddings
    using inner products.
    """
    
    def __init__(self, dropout=0.0):
        """Initialize the inner product decoder.
        
        Args:
            dropout: Dropout probability
        """
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
    
    def forward(self, z, edge_index=None):
        """Forward pass through the decoder.
        
        Args:
            z: Node embeddings of shape [num_nodes, latent_dim]
            edge_index: Optional edge indices for which to compute probabilities
            
        Returns:
            Adjacency matrix reconstructed from inner products
        """
        z = F.dropout(z, p=self.dropout, training=self.training)
        
        if edge_index is None:
            # Compute full adjacency matrix
            adj = torch.matmul(z, z.t())
            return torch.sigmoid(adj)
        else:
            # Compute probabilities only for the given edges
            row, col = edge_index
            return torch.sigmoid(torch.sum(z[row] * z[col], dim=1))

class LocalityPreservingDecoder(nn.Module):
    """Locality preserving decoder implementing the method described in the paper.
    
    This decoder combines inner product with a probability-generating
    function to preserve local connectivity.
    """
    
    def __init__(self, dropout=0.0, max_walk_length=3):
        """Initialize the locality preserving decoder.
        
        Args:
            dropout: Dropout probability
            max_walk_length: Maximum random walk length for PGF
        """
        super(LocalityPreservingDecoder, self).__init__()
        self.dropout = dropout
        self.sparsifier = LocalityPreservingSparsification(max_walk_length)
    
    def forward(self, z, edge_index=None):
        """Forward pass through the decoder.
        
        Args:
            z: Node embeddings of shape [num_nodes, latent_dim]
            edge_index: Optional edge indices for which to compute probabilities
            
        Returns:
            Sparsified adjacency matrix
        """
        z = F.dropout(z, p=self.dropout, training=self.training)
        
        # Apply locality preserving sparsification
        sparsified_similarities = self.sparsifier(z, edge_index)
        
        # Apply sigmoid to get probabilities
        return torch.sigmoid(sparsified_similarities)