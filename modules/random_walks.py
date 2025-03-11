# modules/random_walks.py
import torch
import torch.nn as nn
import numpy as np
import networkx as nx

class RandomWalkProbabilityGenerator(nn.Module):
    """Generate random walk probabilities and probability generating functions.
    
    This module implements the computation of random walk statistics and their
    corresponding probability generating functions (PGF).
    """
    
    def __init__(self, max_walk_length=3):
        """Initialize the random walk probability generator.
        
        Args:
            max_walk_length: Maximum length of random walks to consider
        """
        super(RandomWalkProbabilityGenerator, self).__init__()
        self.max_walk_length = max_walk_length
    
    def compute_transition_matrix(self, edge_index, edge_weight=None, num_nodes=None):
        """Compute the transition matrix for random walks.
        
        Args:
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            num_nodes: Number of nodes in the graph
            
        Returns:
            Transition matrix P
        """
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Compute node degrees
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.index_add_(0, edge_index[0], edge_weight)
        
        # Create sparse transition matrix
        row, col = edge_index
        P = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        
        for i in range(edge_index.size(1)):
            src, dst = row[i].item(), col[i].item()
            weight = edge_weight[i].item()
            P[src, dst] = weight / max(deg[src].item(), 1e-8)
        
        return P
    
    def compute_pgf_moments(self, P, node_idx):
        """Compute moments for the probability generating function.
        
        Args:
            P: Transition matrix
            node_idx: Index of the node
            
        Returns:
            Mean (μ) and variance (σ²) for the PGF
        """
        # Compute expected number of visits using powers of P
        P_powers = [torch.eye(P.size(0), device=P.device)]
        for _ in range(self.max_walk_length):
            P_powers.append(torch.matmul(P_powers[-1], P))
        
        # Compute mean (first moment)
        mu = 0
        for k in range(1, len(P_powers)):
            mu += P_powers[k][node_idx, node_idx]
        
        # Compute second moment
        second_moment = 0
        for k in range(1, len(P_powers)):
            for j in range(1, len(P_powers)):
                if k != j:
                    second_moment += P_powers[k][node_idx, node_idx] * P_powers[j][node_idx, node_idx]
        
        # Compute variance
        variance = second_moment - mu**2 + mu  # Adding mu accounts for diagonal terms
        
        return mu, variance
    
    def compute_pgf(self, mu, sigma_sq, lambda_val=1.0):
        """Compute the probability generating function up to order 3.
        
        Args:
            mu: Mean parameter
            sigma_sq: Variance parameter
            lambda_val: Lambda parameter
            
        Returns:
            PGF value G(λ)
        """
        # G(λ) ≈ 1 + μλ + (1/2)(μ^2 + σ^2)λ^2
        pgf_value = 1 + mu * lambda_val + 0.5 * (mu**2 + sigma_sq) * (lambda_val**2)
        return pgf_value
    
    def forward(self, edge_index, edge_weight=None, num_nodes=None):
        """Compute PGF values for all nodes.
        
        Args:
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            num_nodes: Number of nodes in the graph
            
        Returns:
            Dictionary mapping node indices to (mu, sigma_sq) tuples
        """
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1
        
        # Compute transition matrix
        P = self.compute_transition_matrix(edge_index, edge_weight, num_nodes)
        
        # Compute PGF moments for each node
        pgf_moments = {}
        for node_idx in range(num_nodes):
            mu, sigma_sq = self.compute_pgf_moments(P, node_idx)
            pgf_moments[node_idx] = (mu, sigma_sq)
        
        return pgf_moments