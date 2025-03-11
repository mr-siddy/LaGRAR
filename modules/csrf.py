# modules/csrf.py
import torch
import numpy as np
import networkx as nx
from .curvature import compute_ollivier_ricci_curvature

class CurvatureSensitiveRicciFlow:
    """Implementation of Curvature Sensitive Ricci Flow (CSRF) module.
    
    This module applies Ricci flow operations to transform graph topology
    based on edge curvatures, as described in the paper.
    """
    
    def __init__(self, alpha=0.5, beta=0.5):
        """Initialize CSRF module.
        
        Args:
            alpha: Parameter controlling sensitivity for positive curvature
            beta: Parameter controlling sensitivity for negative curvature
        """
        self.alpha = alpha
        self.beta = beta
    
    def apply_flow(self, edge_index, edge_weight=None, num_nodes=None, iterations=1):
        """Apply Curvature Sensitive Ricci Flow to the graph.
        
        Args:
            edge_index: Tensor of shape [2, num_edges] containing edge indices
            edge_weight: Optional tensor of shape [num_edges] containing edge weights
            num_nodes: Number of nodes in the graph
            iterations: Number of CSRF iterations to perform
            
        Returns:
            Transformed edge_index and edge_weight after CSRF
        """
        # Convert to NetworkX for curvature computation
        edge_list = edge_index.t().cpu().numpy()
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        weights = edge_weight.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        for i, (u, v) in enumerate(edge_list):
            G.add_edge(u, v, weight=weights[i])
        
        for _ in range(iterations):
            # Compute Ollivier-Ricci curvature for all edges
            edge_curvatures = compute_ollivier_ricci_curvature(G)
            
            # Update edge weights according to CSRF formula (Eq. 6 in the paper)
            new_weights = {}
            total_weight = 0
            
            for (u, v), kappa in edge_curvatures.items():
                if kappa >= 0:
                    # Positively curved edges (contraction)
                    new_w = weights[G[u][v]['weight']] * (1 - (self.alpha / max(G.degree(u), G.degree(v))) * np.exp(self.beta * kappa))
                else:
                    # Negatively curved edges (stretching)
                    new_w = weights[G[u][v]['weight']] * (1 + (self.beta / max(G.degree(u), G.degree(v))) * np.exp(self.alpha * kappa))
                
                new_weights[(u, v)] = new_w
                total_weight += new_w
            
            # Update edge weights in the graph
            for (u, v), w in new_weights.items():
                G[u][v]['weight'] = w
        
        # Convert back to PyTorch tensors
        new_edge_index = []
        new_edge_weight = []
        
        for u, v, data in G.edges(data=True):
            new_edge_index.append([u, v])
            new_edge_weight.append(data['weight'])
        
        new_edge_index = torch.tensor(new_edge_index, device=edge_index.device).t()
        new_edge_weight = torch.tensor(new_edge_weight, device=edge_weight.device)
        
        return new_edge_index, new_edge_weight
    
    def __call__(self, graph, iterations=1):
        """Apply CSRF to a graph.
        
        Args:
            graph: Input graph (edge_index, edge_weight, num_nodes)
            iterations: Number of CSRF iterations
            
        Returns:
            Transformed graph after applying CSRF
        """
        edge_index, edge_weight, num_nodes = graph
        return self.apply_flow(edge_index, edge_weight, num_nodes, iterations)