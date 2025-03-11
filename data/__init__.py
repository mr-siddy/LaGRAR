"""
LaGRAR data module - Task-aware Latent Graph Rewiring with Adversarial Ricci flows.

This module contains dataset loading utilities, preprocessing functions, and the implementation
of the TaskGraph synthetic dataset as described in the paper.
"""

from .datasets import (
    load_dataset, 
    get_dataset_stats, 
    get_standard_split, 
    get_homophily_stats
)

from .preprocessing import (
    preprocess_graph, 
    compute_ollivier_ricci_curvature, 
    get_curvature_stats
)

from .taskgraph import (
    generate_homsbm,
    generate_hetsbm,
    generate_rcg, 
    generate_rhg,
    generate_taskgraph
)

from .transforms import (
    CSRFTransform,
    AddSelfLoops,
    NormalizeFeatures
)

__all__ = [
    'load_dataset',
    'get_dataset_stats',
    'get_standard_split',
    'get_homophily_stats',
    'preprocess_graph',
    'compute_ollivier_ricci_curvature',
    'get_curvature_stats',
    'generate_homsbm',
    'generate_hetsbm',
    'generate_rcg',
    'generate_rhg',
    'generate_taskgraph',
    'CSRFTransform',
    'AddSelfLoops',
    'NormalizeFeatures'
]