# modules/locality_preserving.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LocalityPreservingSparsification(nn.Module):
    """Implements the locality preserving sparsification operation using a
    probability-generating function of random walk.
    """
    
    def __init__(self, max_walk_length=3):
        """Initialize the locality preserving sparsification module.
        
        Args:
            max_walk_length: Maximum length of random walks to consider
        """
        super(LocalityPreservingSparsification, self).__init__()
        self.max_walk_length = max_walk_length
    
    def probability_generating_function(self, mu, sigma_sq, lambda_val=1.0):
        """Compute the second-order approximation of the probability generating
        function for random walks, as described in Eq 3 in the paper.
        
        Args:
            mu: Mean parameter
            sigma_sq: Variance parameter
            lambda_val: Lambda parameter
            
        Returns:
            Approximated PGF value
        """
        # G_i^(3)(λ) ≈ 1 + μλ + (1/2)(μ^2 + σ^2)λ^2
        return 1 + mu * lambda_val + 0.5 * (mu**2 + sigma_sq) * lambda_val**2
    
    def forward(self, z, edge_index=None):
        """Apply locality preserving sparsification to inner products.
        
        Args:
            z: Node embeddings of shape [num_nodes, hidden_dim]
            edge_index: Optional edge indices for which to compute sparsification
            
        Returns:
            Sparsified similarity matrix or edge weights
        """
        num_nodes = z.size(0)
        
        # Normalize embeddings for stable dot products
        z_norm = F.normalize(z, p=2, dim=1)
        
        if edge_index is None:
            # Compute all pairwise inner products
            similarities = torch.matmul(z_norm, z_norm.t())
            
            # Compute PGF values for each node
            mu = z.mean(dim=1)
            sigma_sq = z.var(dim=1)
            
            # Compute PGF values for each pair of nodes
            pgf_values = torch.zeros_like(similarities)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    pgf_values[i, j] = self.probability_generating_function(mu[j], sigma_sq[j])
            
            # Apply sparsification by multiplying with PGF values
            sparsified_similarities = similarities * pgf_values
            
            return sparsified_similarities
        else:
            # Compute inner products only for the given edges
            src, dst = edge_index
            z_src = z_norm[src]
            z_dst = z_norm[dst]
            
            # Compute pairwise similarities
            similarities = (z_src * z_dst).sum(dim=1)
            
            # Compute PGF values for destination nodes
            mu = z.mean(dim=1)
            sigma_sq = z.var(dim=1)
            
            pgf_values = torch.zeros_like(similarities)
            for i in range(edge_index.size(1)):
                j = dst[i]
                pgf_values[i] = self.probability_generating_function(mu[j], sigma_sq[j])
            
            # Apply sparsification
            sparsified_weights = similarities * pgf_values
            
            return sparsified_weights