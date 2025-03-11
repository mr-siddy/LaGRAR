"""
Dataset loading utilities for LaGRAR.

This module provides functions to load benchmark datasets used in the LaGRAR paper:
1. WebKB datasets (Cornell, Texas, Wisconsin) - small heterophilic datasets
2. WikipediaNetwork (Chameleon, Squirrel, Actor) - large heterophilic datasets
3. Planetoid (Cora, Citeseer, Pubmed) - homophilic datasets
4. TUDatasets (REDDIT-B, IMDB-B, COLLAB, MUTAG, PROTEINS, ENZYMES) - graph classification
"""

import os
import torch
import numpy as np
import os.path as osp
from typing import Tuple, List, Dict, Optional, Union

import torch_geometric
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid, TUDataset
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.transforms import NormalizeFeatures

# Define dataset groups as in the paper
SHD_DATASETS = ['cornell', 'wisconsin', 'texas']  # Small heterophilic datasets
MHD_DATASETS = ['chameleon', 'squirrel', 'actor']  # Large heterophilic datasets
MHOD_DATASETS = ['cora', 'citeseer', 'pubmed']  # Homophilic datasets
GC_DATASETS = ['REDDIT-BINARY', 'IMDB-BINARY', 'COLLAB', 'MUTAG', 'PROTEINS', 'ENZYMES']  # Graph classification datasets

def load_dataset(name: str, root: str = 'data', transform=None) -> Union[Data, InMemoryDataset]:
    """
    Load one of the benchmark datasets used in LaGRAR paper.
    
    Args:
        name: Name of the dataset (case-insensitive)
        root: Root directory for dataset storage
        transform: Optional transforms to apply
    
    Returns:
        PyG dataset or data object
    """
    name_lower = name.lower()
    
    # Default transform for feature normalization
    if transform is None:
        transform = NormalizeFeatures()
    
    # Small heterophilic datasets (WebKB)
    if name_lower in ['cornell', 'texas', 'wisconsin']:
        return WebKB(root=root, name=name.capitalize(), transform=transform)
    
    # Large heterophilic datasets (WikipediaNetwork)
    elif name_lower in ['chameleon', 'squirrel']:
        return WikipediaNetwork(root=root, name=name.capitalize(), transform=transform)
    elif name_lower == 'actor':
        return Actor(root=root, transform=transform)
    
    # Homophilic datasets (Planetoid)
    elif name_lower in ['cora', 'citeseer', 'pubmed']:
        return Planetoid(root=root, name=name.capitalize(), transform=transform)
    
    # Graph classification datasets (TUDataset)
    elif name.upper() in GC_DATASETS:
        return TUDataset(root=root, name=name.upper(), transform=transform)
    
    else:
        raise ValueError(f"Dataset {name} not supported. Available datasets: "
                         f"{SHD_DATASETS + MHD_DATASETS + MHOD_DATASETS + GC_DATASETS}")

def get_dataset_stats(dataset) -> Dict:
    """
    Get statistics for a given dataset as reported in the paper (Table 8).
    
    Args:
        dataset: PyG dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    if isinstance(dataset, InMemoryDataset) and hasattr(dataset, 'data'):
        # For node classification datasets
        data = dataset.data
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1) // 2  # Undirected graph
        feature_dim = data.x.size(1)
        if hasattr(data, 'y'):
            num_classes = len(torch.unique(data.y))
        else:
            num_classes = 0
            
        return {
            'name': dataset.name,
            '|V|': num_nodes,
            '|E|': num_edges,
            'd': feature_dim,
            'C': num_classes
        }
    
    elif hasattr(dataset, 'num_features') and hasattr(dataset, 'num_classes'):
        # For graph classification datasets
        num_graphs = len(dataset)
        avg_nodes = sum(data.num_nodes for data in dataset) / num_graphs
        avg_edges = sum(data.edge_index.size(1) // 2 for data in dataset) / num_graphs
        
        return {
            'name': dataset.name,
            'num_graphs': num_graphs,
            'avg_nodes': avg_nodes,
            'avg_edges': avg_edges,
            'd': dataset.num_features,
            'C': dataset.num_classes
        }
    
    else:
        raise ValueError("Unsupported dataset type")

def get_homophily_stats(data) -> float:
    """
    Calculate the homophily ratio H(G) for a graph as reported in the paper (Table 8).
    
    Args:
        data: PyG data object
    
    Returns:
        Homophily ratio (0 to 1)
    """
    edge_index = data.edge_index
    labels = data.y
    
    # Get source and destination nodes
    src, dst = edge_index
    
    # Count edges connecting same-label nodes
    same_label_edges = (labels[src] == labels[dst]).sum().item()
    
    # Total number of edges
    total_edges = edge_index.size(1)
    
    # Calculate homophily ratio
    homophily = same_label_edges / total_edges
    
    return homophily

def get_standard_split(dataset, task='node_classification', val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Create standard train/val/test splits for datasets.
    
    Args:
        dataset: PyG dataset
        task: Task type ('node_classification', 'link_prediction', 'graph_classification')
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    
    Returns:
        Split indices according to the task
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if task == 'node_classification':
        # Use existing splits if available
        if hasattr(dataset.data, 'train_mask') and hasattr(dataset.data, 'val_mask') and hasattr(dataset.data, 'test_mask'):
            return {
                'train_idx': torch.nonzero(dataset.data.train_mask, as_tuple=True)[0],
                'val_idx': torch.nonzero(dataset.data.val_mask, as_tuple=True)[0],
                'test_idx': torch.nonzero(dataset.data.test_mask, as_tuple=True)[0]
            }
        
        # Create new splits
        num_nodes = dataset.data.x.size(0)
        indices = torch.randperm(num_nodes)
        
        test_size = int(num_nodes * test_ratio)
        val_size = int(num_nodes * val_ratio)
        train_size = num_nodes - val_size - test_size
        
        return {
            'train_idx': indices[:train_size],
            'val_idx': indices[train_size:train_size + val_size],
            'test_idx': indices[train_size + val_size:]
        }
    
    elif task == 'link_prediction':
        edge_index = dataset.data.edge_index
        num_edges = edge_index.size(1) // 2  # Undirected edges
        
        # Get upper triangular edge indices (to avoid duplicates)
        edge_list = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src < dst:  # Upper triangular
                edge_list.append((src, dst))
        
        # Shuffle edges
        np.random.shuffle(edge_list)
        edge_list = np.array(edge_list)
        
        # Create positive samples
        test_size = int(len(edge_list) * test_ratio)
        val_size = int(len(edge_list) * val_ratio)
        train_size = len(edge_list) - val_size - test_size
        
        train_pos = edge_list[:train_size]
        val_pos = edge_list[train_size:train_size + val_size]
        test_pos = edge_list[train_size + val_size:]
        
        # Create negative samples (random non-edges)
        neg_samples = []
        n = dataset.data.num_nodes
        edge_set = set([(src, dst) for src, dst in edge_list])
        edge_set.update([(dst, src) for src, dst in edge_list])  # Add reverse edges
        
        while len(neg_samples) < len(edge_list):
            i, j = np.random.randint(0, n, 2)
            if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
                neg_samples.append((i, j))
                edge_set.add((i, j))
                edge_set.add((j, i))
        
        neg_samples = np.array(neg_samples)
        
        train_neg = neg_samples[:train_size]
        val_neg = neg_samples[train_size:train_size + val_size]
        test_neg = neg_samples[train_size + val_size:]
        
        return {
            'train_pos': torch.tensor(train_pos, dtype=torch.long).t(),
            'val_pos': torch.tensor(val_pos, dtype=torch.long).t(),
            'test_pos': torch.tensor(test_pos, dtype=torch.long).t(),
            'train_neg': torch.tensor(train_neg, dtype=torch.long).t(),
            'val_neg': torch.tensor(val_neg, dtype=torch.long).t(),
            'test_neg': torch.tensor(test_neg, dtype=torch.long).t()
        }
    
    elif task == 'graph_classification':
        num_graphs = len(dataset)
        indices = torch.randperm(num_graphs)
        
        test_size = int(num_graphs * test_ratio)
        val_size = int(num_graphs * val_ratio)
        train_size = num_graphs - val_size - test_size
        
        return {
            'train_idx': indices[:train_size],
            'val_idx': indices[train_size:train_size + val_size],
            'test_idx': indices[train_size + val_size:]
        }
    
    else:
        raise ValueError(f"Unsupported task: {task}")