"""
Implementation of TaskGraph synthetic dataset as described in the paper.

The TaskGraph dataset includes four types of graphs:
1. Homophilic Community (Hom-SBM): Stochastic Block Model with homophilic feature distribution
2. Heterophilic Community (Het-SBM): Stochastic Block Model with heterophilic feature distribution
3. Random Cyclic Geometry (RCG): Heterophilic graph with cyclic structure 
4. Random Hyperbolic Geometry (RHG): Heterophilic graph with hyperbolic structure

This implementation follows the specifications in Section A.7 of the paper.
"""

import torch
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional, Union
from scipy import stats

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx

class TaskGraph(InMemoryDataset):
    """
    TaskGraph dataset as described in the LaGRAR paper.
    
    Args:
        root: Root directory for dataset storage
        name: Type of graph ('homsbm', 'hetsbm', 'rcg', 'rhg')
        num_nodes: Number of nodes per class
        num_classes: Number of classes
        transform: Optional transforms
        pre_transform: Optional pre-transforms
    """
    
    def __init__(self, root: str, name: str, num_nodes: int = 200, num_classes: int = 2, 
                 transform=None, pre_transform=None):
        self.name = name.lower()
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        
        if self.name not in ['homsbm', 'hetsbm', 'rcg', 'rhg']:
            raise ValueError(f"Unknown graph type: {name}. Available types: homsbm, hetsbm, rcg, rhg")
        
        super(TaskGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [f'{self.name}.pt']
    
    @property
    def processed_file_names(self):
        return [f'{self.name}_processed.pt']
    
    def download(self):
        # Generate the graph data
        if self.name == 'homsbm':
            data = generate_homsbm(self.num_nodes, self.num_classes)
        elif self.name == 'hetsbm':
            data = generate_hetsbm(self.num_nodes, self.num_classes)
        elif self.name == 'rcg':
            data = generate_rcg(self.num_nodes, self.num_classes)
        else:  # self.name == 'rhg'
            data = generate_rhg(self.num_nodes, self.num_classes)
        
        # Save the raw data
        torch.save(data, self.raw_paths[0])
    
    def process(self):
        # Load raw data
        data = torch.load(self.raw_paths[0])
        
        # Apply pre-transform if available
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        # Process the data
        data_list = [data]
        
        # Save the processed data
        torch.save(self.collate(data_list), self.processed_paths[0])

def generate_taskgraph(task_type: str, num_nodes_per_class: int = 200, num_classes: int = 2,
                    feature_dim: int = 64, seed: int = 42) -> Data:
    """
    Generate a TaskGraph as described in the paper (Section A.7).
    
    Args:
        task_type: Type of graph ('homsbm', 'hetsbm', 'rcg', 'rhg')
        num_nodes_per_class: Number of nodes per class
        num_classes: Number of classes
        feature_dim: Feature dimension
        seed: Random seed
    
    Returns:
        PyG Data object
    """
    if task_type.lower() == 'homsbm':
        return generate_homsbm(num_nodes_per_class, num_classes, feature_dim=feature_dim, seed=seed)
    elif task_type.lower() == 'hetsbm':
        return generate_hetsbm(num_nodes_per_class, num_classes, feature_dim=feature_dim, seed=seed)
    elif task_type.lower() == 'rcg':
        return generate_rcg(num_nodes_per_class, num_classes, feature_dim=feature_dim, seed=seed)
    elif task_type.lower() == 'rhg':
        return generate_rhg(num_nodes_per_class, num_classes, feature_dim=feature_dim, seed=seed)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def generate_homsbm(num_nodes_per_class: int = 200, num_classes: int = 2, 
                    p_intra: float = 0.05, p_inter: float = 0.01, 
                    feature_dim: int = 64, seed: int = 42) -> Data:
    """
    Generate Homophilic Community (Hom-SBM) graph as described in the paper.
    
    As specified in Section A.7: "Hom-SBM featured an intra-community connection 
    probability of 0.05 and an inter-community probability of 0.01, emphasizing sparse but
    homophilic interactions. Node features are generated from a normal distribution
    with means of 0 for the first community and 1 for the second, both with a standard deviation of 1.0."
    
    Args:
        num_nodes_per_class: Number of nodes per class
        num_classes: Number of classes
        p_intra: Probability of intra-community edges
        p_inter: Probability of inter-community edges
        feature_dim: Dimension of node features
        seed: Random seed
    
    Returns:
        PyG Data object
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Total number of nodes
    num_nodes = num_nodes_per_class * num_classes
    
    # Generate SBM graph
    sizes = [num_nodes_per_class] * num_classes  # Equal-sized communities
    probs = np.ones((num_classes, num_classes)) * p_inter  # Inter-community edge probability
    np.fill_diagonal(probs, p_intra)  # Intra-community edge probability
    
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    
    # Generate node features (homophilic)
    x = torch.zeros((num_nodes, feature_dim))
    for c in range(num_classes):
        # Nodes in class c
        start_idx = c * num_nodes_per_class
        end_idx = (c + 1) * num_nodes_per_class
        
        # Generate features from normal distribution with mean based on class
        # As mentioned in the paper, nodes in first community have mean 0, second have mean 1
        mean = c
        std = 1.0
        x[start_idx:end_idx] = torch.normal(mean, std, size=(num_nodes_per_class, feature_dim))
    
    # Generate node labels
    y = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start_idx = c * num_nodes_per_class
        end_idx = (c + 1) * num_nodes_per_class
        y[start_idx:end_idx] = c
    
    # Convert NetworkX graph to PyG data object
    data = from_networkx(G)
    data.x = x
    data.y = y
    
    return data

def generate_hetsbm(num_nodes_per_class: int = 200, num_classes: int = 2, 
                    p_intra: float = 0.01, p_inter: float = 0.05, 
                    feature_dim: int = 64, seed: int = 42) -> Data:
    """
    Generate Heterophilic Community (Het-SBM) graph as described in the paper.
    
    As specified in Section A.7: "Het-SBM utilizes a contrasting setup with an intra-community probability
    of 0.01 and an inter-community probability of 0.05 to simulate sparse intra-community and denser
    inter-community connections. Node features are drawn from a uniform distribution with the first community
    features ranging from 0 to 1 and the second from 5 to 10, illustrating significant feature variance
    between communities."
    
    Args:
        num_nodes_per_class: Number of nodes per class
        num_classes: Number of classes
        p_intra: Probability of intra-community edges
        p_inter: Probability of inter-community edges
        feature_dim: Dimension of node features
        seed: Random seed
    
    Returns:
        PyG Data object
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Total number of nodes
    num_nodes = num_nodes_per_class * num_classes
    
    # Generate SBM graph (heterophilic: higher inter-community probability)
    sizes = [num_nodes_per_class] * num_classes
    probs = np.ones((num_classes, num_classes)) * p_inter  # Higher inter-community edge probability
    np.fill_diagonal(probs, p_intra)  # Lower intra-community edge probability
    
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    
    # Generate node features (heterophilic)
    x = torch.zeros((num_nodes, feature_dim))
    for c in range(num_classes):
        # Nodes in class c
        start_idx = c * num_nodes_per_class
        end_idx = (c + 1) * num_nodes_per_class
        
        # Uniform distribution with different ranges for each community
        if c == 0:
            # First community: features from 0 to 1
            x[start_idx:end_idx] = torch.rand((num_nodes_per_class, feature_dim))
        else:
            # Second community: features from 5 to 10
            x[start_idx:end_idx] = 5 + 5 * torch.rand((num_nodes_per_class, feature_dim))
    
    # Generate node labels
    y = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start_idx = c * num_nodes_per_class
        end_idx = (c + 1) * num_nodes_per_class
        y[start_idx:end_idx] = c
    
    # Convert NetworkX graph to PyG data object
    data = from_networkx(G)
    data.x = x
    data.y = y
    
    return data

def generate_rcg(num_nodes_per_class: int = 200, num_classes: int = 2,
                feature_dim: int = 64, seed: int = 42) -> Data:
    """
    Generate Random Cyclic Geometry (RCG) graph as described in the paper.
    
    As specified in Section A.7: "RCG uses a beta distribution with shape parameters of (2, 5) for nodes in
    the first community and (5, 2) for the second, capturing asymmetric feature distributions that reflect
    cyclic interaction patterns."
    
    Args:
        num_nodes_per_class: Number of nodes per class
        num_classes: Number of classes
        feature_dim: Dimension of node features
        seed: Random seed
    
    Returns:
        PyG Data object
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Total number of nodes
    num_nodes = num_nodes_per_class * num_classes
    
    # Generate a cyclic graph structure
    G = nx.cycle_graph(num_nodes)
    
    # Add random edges to create more complex cyclic structures
    # Adding shortcuts through the cycle to create random cyclic geometry
    for i in range(num_nodes):
        # Add random connections (shortcuts)
        num_shortcuts = np.random.randint(1, 4)  # 1-3 random shortcuts
        possible_targets = list(range(num_nodes))
        possible_targets.remove(i)
        if i > 0:
            possible_targets.remove(i - 1)  # Remove existing neighbors
        if i < num_nodes - 1:
            possible_targets.remove(i + 1)  # Remove existing neighbors
        
        # If node is at the end of the cycle
        if i == num_nodes - 1:
            possible_targets.remove(0)  # Remove wrapped connection
        
        targets = np.random.choice(possible_targets, 
                                   size=min(num_shortcuts, len(possible_targets)), 
                                   replace=False)
        for target in targets:
            G.add_edge(i, target)
    
    # Generate node features using beta distributions
    x = torch.zeros((num_nodes, feature_dim))
    for c in range(num_classes):
        # Nodes in class c
        start_idx = c * num_nodes_per_class
        end_idx = (c + 1) * num_nodes_per_class
        
        # Beta distribution with different parameters for each community
        if c == 0:
            # First community: Beta(2, 5)
            beta_samples = np.random.beta(2, 5, size=(num_nodes_per_class, feature_dim))
            x[start_idx:end_idx] = torch.tensor(beta_samples, dtype=torch.float)
        else:
            # Second community: Beta(5, 2)
            beta_samples = np.random.beta(5, 2, size=(num_nodes_per_class, feature_dim))
            x[start_idx:end_idx] = torch.tensor(beta_samples, dtype=torch.float)
    
    # Generate node labels
    y = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start_idx = c * num_nodes_per_class
        end_idx = (c + 1) * num_nodes_per_class
        y[start_idx:end_idx] = c
    
    # Convert NetworkX graph to PyG data object
    data = from_networkx(G)
    data.x = x
    data.y = y
    
    return data

def generate_rhg(num_nodes_per_class: int = 200, num_classes: int = 2,
                feature_dim: int = 64, seed: int = 42) -> Data:
    """
    Generate Random Hyperbolic Geometry (RHG) graph as described in the paper.
    
    As specified in Section A.7: "RHG implements an exponential distribution with scale parameters
    of 1 for the first community and 0.5 for the second, aligning with the hyperbolic space's hierarchical
    nature."
    
    Args:
        num_nodes_per_class: Number of nodes per class
        num_classes: Number of classes
        feature_dim: Dimension of node features
        seed: Random seed
    
    Returns:
        PyG Data object
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Total number of nodes
    num_nodes = num_nodes_per_class * num_classes
    
    # Create a tree-like structure to simulate hyperbolic geometry
    # Starting with a random tree
    G = nx.random_tree(num_nodes, seed=seed)
    
    # Add some edges to create cycles and complex structures while maintaining
    # tree-like (negative curvature) properties overall
    for i in range(num_nodes // 10):  # Add a limited number of extra edges
        source = np.random.randint(0, num_nodes)
        target = np.random.randint(0, num_nodes)
        if source != target and not G.has_edge(source, target):
            G.add_edge(source, target)
    
    # Generate node features using exponential distributions
    x = torch.zeros((num_nodes, feature_dim))
    for c in range(num_classes):
        # Nodes in class c
        start_idx = c * num_nodes_per_class
        end_idx = (c + 1) * num_nodes_per_class
        
        # Exponential distribution with different scale parameters for each community
        if c == 0:
            # First community: scale parameter 1.0
            exp_samples = np.random.exponential(scale=1.0, size=(num_nodes_per_class, feature_dim))
            x[start_idx:end_idx] = torch.tensor(exp_samples, dtype=torch.float)
        else:
            # Second community: scale parameter 0.5
            exp_samples = np.random.exponential(scale=0.5, size=(num_nodes_per_class, feature_dim))
            x[start_idx:end_idx] = torch.tensor(exp_samples, dtype=torch.float)
    
    # Generate node labels
    y = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start_idx = c * num_nodes_per_class
        end_idx = (c + 1) * num_nodes_per_class
        y[start_idx:end_idx] = c
    
    # Convert NetworkX graph to PyG data object
    data = from_networkx(G)
    data.x = x
    data.y = y
    
    return data