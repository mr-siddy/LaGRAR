"""
Data preprocessing utilities for LaGRAR.

This module provides functions for graph preprocessing, computing Ollivier-Ricci curvature,
and other utilities needed for LaGRAR implementation.
"""

import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import Tuple, Dict, List, Optional, Union

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, to_scipy_sparse_matrix, from_scipy_sparse_matrix

# Try to import GraphRicciCurvature package, provide backup implementation if not available
try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    HAS_RICCI_PACKAGE = True
except ImportError:
    HAS_RICCI_PACKAGE = False

def preprocess_graph(data: Data) -> Data:
    """
    Preprocess a PyG graph for LaGRAR.
    
    Args:
        data: PyG data object
    
    Returns:
        Preprocessed PyG data object
    """
    # Make a copy to avoid modifying the original
    processed_data = Data()
    for key, item in data:
        processed_data[key] = item
    
    # Add self-loops if not present
    edge_index = processed_data.edge_index
    num_nodes = processed_data.num_nodes
    
    # Check for self-loops
    has_self_loops = torch.any((edge_index[0] == edge_index[1]))
    
    if not has_self_loops:
        # Add self-loops
        diagonal_idx = torch.arange(num_nodes)
        self_loops = torch.stack([diagonal_idx, diagonal_idx], dim=0)
        
        # Concatenate with existing edges
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        processed_data.edge_index = edge_index
    
    # Create edge weights if not present
    if not hasattr(processed_data, 'edge_weight') or processed_data.edge_weight is None:
        processed_data.edge_weight = torch.ones(processed_data.edge_index.size(1))
    
    return processed_data

def compute_ollivier_ricci_curvature(data: Data, alpha: float = 0.5, weight_attr: str = 'weight') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Ollivier-Ricci curvature for each edge in the graph as used in the paper (Eq. 11).
    
    Args:
        data: PyG data object
        alpha: ORC parameter alpha (probability to stay at the same node during random walk)
        weight_attr: Edge attribute name for weights
    
    Returns:
        edge_index: Edge index tensor
        edge_curvature: Corresponding ORC values tensor
    """
    # Convert to NetworkX for curvature computation
    G = to_networkx(data, to_undirected=True, edge_attrs=[weight_attr] if hasattr(data, weight_attr) else None)
    
    # Initialize curvature values
    if HAS_RICCI_PACKAGE:
        # Use GraphRicciCurvature package
        orc = OllivierRicci(G, alpha=alpha, verbose="INFO")
        G = orc.compute_ricci_curvature()
        
        # Extract curvature values
        edge_curvatures = []
        edge_list = []
        
        for src, dst, data in G.edges(data=True):
            edge_list.append((src, dst))
            edge_curvatures.append(data['ricciCurvature'])
            
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_curvature = torch.tensor(edge_curvatures, dtype=torch.float)
        
    else:
        # Backup implementation using discrete ORC computation
        edge_index = data.edge_index
        edge_curvature = torch.zeros(edge_index.size(1))
        
        # For each edge, compute ORC using the formula from the paper (Eq. 11)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            # Get neighbor indices
            src_neighbors = edge_index[1, edge_index[0] == src].tolist()
            dst_neighbors = edge_index[1, edge_index[0] == dst].tolist()
            
            # Compute random walk probability measures
            src_measure = {n: (1-alpha)/len(src_neighbors) for n in src_neighbors}
            src_measure[src] = alpha  # Add self probability
            
            dst_measure = {n: (1-alpha)/len(dst_neighbors) for n in dst_neighbors}
            dst_measure[dst] = alpha  # Add self probability
            
            # Compute earth mover distance (approximation)
            # Simplified Wasserstein distance computation (not optimal but gives an approximation)
            W1 = 0
            common_neighbors = set(src_neighbors) & set(dst_neighbors)
            all_neighbors = set(src_neighbors) | set(dst_neighbors) | {src, dst}
            
            # Distance matrix (shortest path distances)
            dist = {n: {} for n in all_neighbors}
            
            # Compute distances for each pair of nodes
            for n1 in all_neighbors:
                for n2 in all_neighbors:
                    if n1 == n2:
                        dist[n1][n2] = 0
                    elif n2 in src_neighbors and n1 in dst_neighbors:
                        dist[n1][n2] = 2  # 2-hop distance
                    else:
                        dist[n1][n2] = 1  # Direct neighbors
            
            # Simple greedy transport (not optimal)
            remaining_src = src_measure.copy()
            remaining_dst = dst_measure.copy()
            
            for n1 in all_neighbors:
                if n1 in remaining_src and remaining_src[n1] > 0:
                    for n2 in all_neighbors:
                        if n2 in remaining_dst and remaining_dst[n2] > 0:
                            flow = min(remaining_src[n1], remaining_dst[n2])
                            W1 += flow * dist[n1][n2]
                            remaining_src[n1] -= flow
                            remaining_dst[n2] -= flow
                            if remaining_src[n1] <= 1e-10:
                                break
            
            # Compute curvature using the formula κ(x,y) = 1 - W1(μx,μy)/d(x,y)
            edge_curvature[i] = 1 - W1
    
    return edge_index, edge_curvature

def get_curvature_stats(edge_curvature: torch.Tensor) -> Dict[str, float]:
    """
    Get curvature statistics as reported in the paper (Table 8).
    
    Args:
        edge_curvature: Tensor of edge curvature values
    
    Returns:
        Dictionary with curvature statistics
    """
    return {
        'ORC Mean': edge_curvature.mean().item(),
        'ORC Min': edge_curvature.min().item(),
        'ORC Max': edge_curvature.max().item(),
        'ORC Std': edge_curvature.std().item(),
        'Positive Curvature (%)': 100 * (edge_curvature > 0).sum().item() / len(edge_curvature),
        'Negative Curvature (%)': 100 * (edge_curvature < 0).sum().item() / len(edge_curvature),
        'Zero Curvature (%)': 100 * (edge_curvature == 0).sum().item() / len(edge_curvature)
    }

def apply_curvature_sensitive_ricci_flow(data: Data, iterations: int = 1, alpha: float = 0.5, beta: float = 0.5) -> Data:
    """
    Apply Curvature Sensitive Ricci Flow (CSRF) to the graph as described in the paper (Eq. 6).
    
    Args:
        data: PyG data object
        iterations: Number of CSRF iterations
        alpha: Ricci flow parameter for negative curvature
        beta: Ricci flow parameter for positive curvature
    
    Returns:
        PyG data object with updated edge weights
    """
    # Make a copy to avoid modifying the original
    processed_data = preprocess_graph(data)
    edge_index, edge_weight = processed_data.edge_index, processed_data.edge_weight
    
    # Get node degrees
    degrees = torch.bincount(edge_index[0], minlength=processed_data.num_nodes)
    
    for _ in range(iterations):
        # Compute Ollivier-Ricci curvature
        _, edge_curvature = compute_ollivier_ricci_curvature(processed_data)
        
        # Apply CSRF weight updates as per Eq. 6
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            k = edge_curvature[i].item()
            deg_src, deg_dst = degrees[src].item(), degrees[dst].item()
            max_deg = max(deg_src, deg_dst)
            
            if k >= 0:
                # Positive curvature case (contract)
                edge_weight[i] = edge_weight[i] * (1 - alpha/max_deg * np.exp(beta * k))
            else:
                # Negative curvature case (expand)
                edge_weight[i] = edge_weight[i] * (1 + beta/max_deg * np.exp(alpha * k))
        
        # Normalize weights to conserve total edge weights
        edge_weight = edge_weight / edge_weight.mean() * 1.0
        processed_data.edge_weight = edge_weight
    
    return processed_data

def create_sparse_adjacency(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> torch.sparse.FloatTensor:
    """
    Create sparse adjacency matrix from edge index and weights.
    
    Args:
        edge_index: Edge index tensor
        edge_weight: Edge weight tensor
        num_nodes: Number of nodes
    
    Returns:
        Sparse adjacency tensor
    """
    # Create sparse adjacency matrix
    return torch.sparse.FloatTensor(
        edge_index, 
        edge_weight, 
        torch.Size([num_nodes, num_nodes])
    )

def get_normalized_adjacency(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> torch.sparse.FloatTensor:
    """
    Create normalized adjacency matrix D^(-1/2) A D^(-1/2) as used in GCN.
    
    Args:
        edge_index: Edge index tensor
        edge_weight: Edge weight tensor
        num_nodes: Number of nodes
    
    Returns:
        Normalized sparse adjacency tensor
    """
    # Create sparse adjacency matrix
    adj = create_sparse_adjacency(edge_index, edge_weight, num_nodes)
    
    # Compute degree matrix D
    degrees = torch.sparse.sum(adj, dim=1).to_dense()
    
    # Compute D^(-1/2)
    d_inv_sqrt = torch.pow(degrees, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    
    # Create diagonal matrix D^(-1/2)
    d_inv_sqrt_matrix = torch.sparse.FloatTensor(
        torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0),
        d_inv_sqrt,
        torch.Size([num_nodes, num_nodes])
    )
    
    # Compute D^(-1/2) A D^(-1/2)
    normalized_adj = torch.sparse.mm(torch.sparse.mm(d_inv_sqrt_matrix, adj), d_inv_sqrt_matrix)
    
    return normalized_adj