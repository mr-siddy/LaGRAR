"""
Graph transforms for LaGRAR implementation.

This module defines PyTorch Geometric transforms used in the LaGRAR model including:
1. CSRFTransform: Curvature Sensitive Ricci Flow transform (Eq. 6 in the paper)
2. AddSelfLoops: Add self-loops to graphs
3. NormalizeFeatures: Normalize node features
"""

import torch
import networkx as nx
import numpy as np
from typing import Tuple, Optional, Union

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx, add_self_loops, remove_self_loops
from torch_geometric.data import Data

from .preprocessing import compute_ollivier_ricci_curvature

class CSRFTransform(BaseTransform):
    """
    Curvature Sensitive Ricci Flow transform as described in the paper (Eq. 6).
    
    Args:
        iterations: Number of Ricci flow iterations
        alpha: CSRF parameter for negative curvature (default: 0.5)
        beta: CSRF parameter for positive curvature (default: 0.5)
        normalized: Whether to return normalized edge weights
        cache_curvature: Whether to cache curvature values in data.edge_curvature
    """
    
    def __init__(self, iterations: int = 1, alpha: float = 0.5, beta: float = 0.5, 
                 normalized: bool = True, cache_curvature: bool = True):
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.normalized = normalized
        self.cache_curvature = cache_curvature
    
    def __call__(self, data: Data) -> Data:
        # Make a copy to avoid modifying the original
        transformed_data = data.clone()
        
        # Add edge weights if not present
        if not hasattr(transformed_data, 'edge_weight') or transformed_data.edge_weight is None:
            transformed_data.edge_weight = torch.ones(transformed_data.edge_index.size(1))
        
        # Get edge index and weights
        edge_index, edge_weight = transformed_data.edge_index, transformed_data.edge_weight
        
        # Get node degrees
        node_degrees = torch.zeros(transformed_data.num_nodes, dtype=torch.long)
        unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
        node_degrees[unique_nodes] = counts
        
        for _ in range(self.iterations):
            # Compute Ollivier-Ricci curvature
            _, edge_curvature = compute_ollivier_ricci_curvature(transformed_data)
            
            # Apply CSRF updates as per Eq. 6
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                k = edge_curvature[i].item()
                max_deg = max(node_degrees[src].item(), node_degrees[dst].item())
                
                if k >= 0:
                    # Positive curvature case (contract)
                    edge_weight[i] = edge_weight[i] * (1 - (self.alpha / max_deg) * np.exp(self.beta * k))
                else:
                    # Negative curvature case (stretch)
                    edge_weight[i] = edge_weight[i] * (1 + (self.beta / max_deg) * np.exp(self.alpha * k))
            
            # Normalize weights to conserve total edge weights
            if self.normalized:
                edge_weight = edge_weight / edge_weight.mean() * 1.0
            
            # Update edge weights
            transformed_data.edge_weight = edge_weight
        
        # Cache curvature values
        if self.cache_curvature:
            _, edge_curvature = compute_ollivier_ricci_curvature(transformed_data)
            transformed_data.edge_curvature = edge_curvature
        
        return transformed_data

class AddSelfLoops(BaseTransform):
    """
    Add self-loops to the graph with optional weight.
    
    Args:
        fill_value: Value for self-loop edge weights
    """
    
    def __init__(self, fill_value: float = 1.0):
        self.fill_value = fill_value
    
    def __call__(self, data: Data) -> Data:
        # Make a copy to avoid modifying the original
        transformed_data = data.clone()
        
        # Add self-loops
        edge_index, edge_weight = add_self_loops(
            transformed_data.edge_index, 
            edge_attr=transformed_data.edge_weight if hasattr(transformed_data, 'edge_weight') else None,
            fill_value=self.fill_value,
            num_nodes=transformed_data.num_nodes
        )
        
        transformed_data.edge_index = edge_index
        if edge_weight is not None:
            transformed_data.edge_weight = edge_weight
        
        return transformed_data

class NormalizeFeatures(BaseTransform):
    """
    Normalize node features to have zero mean and unit variance.
    
    Args:
        mean: Whether to center features to have zero mean
        std: Whether to scale features to have unit variance
    """
    
    def __init__(self, mean: bool = True, std: bool = True):
        self.mean = mean
        self.std = std
    
    def __call__(self, data: Data) -> Data:
        # Make a copy to avoid modifying the original
        transformed_data = data.clone()
        
        if hasattr(transformed_data, 'x') and transformed_data.x is not None:
            x = transformed_data.x
            
            if self.mean:
                x = x - x.mean(dim=0, keepdim=True)
            
            if self.std:
                x = x / (x.std(dim=0, keepdim=True) + 1e-8)
            
            transformed_data.x = x
        
        return transformed_data