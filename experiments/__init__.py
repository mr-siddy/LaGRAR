"""
LaGRAR experiment module - Task-aware Latent Graph Rewiring with Adversarial Ricci flows.

This module contains experiment pipelines for evaluating LaGRAR on various tasks:
1. Node classification
2. Link prediction
3. Graph classification
4. Ablation studies

The experiments follow the setup described in Section 5 and 6 of the paper.
"""

from .config import (
    NODE_CLASSIFICATION_CONFIGS,
    LINK_PREDICTION_CONFIGS,
    GRAPH_CLASSIFICATION_CONFIGS,
    SHD_DATASETS,
    MHD_DATASETS, 
    MHOD_DATASETS,
    GC_DATASETS
)

from .node_classification import (
    run_node_classification_experiment,
    evaluate_node_classification
)

from .link_prediction import (
    run_link_prediction_experiment,
    evaluate_link_prediction
)

from .graph_classification import (
    run_graph_classification_experiment,
    evaluate_graph_classification
)

from .ablation_study import (
    run_ablation_study,
    ablation_components
)

__all__ = [
    # Configuration
    'NODE_CLASSIFICATION_CONFIGS',
    'LINK_PREDICTION_CONFIGS',
    'GRAPH_CLASSIFICATION_CONFIGS',
    'SHD_DATASETS',
    'MHD_DATASETS',
    'MHOD_DATASETS',
    'GC_DATASETS',
    
    # Node classification
    'run_node_classification_experiment',
    'evaluate_node_classification',
    
    # Link prediction
    'run_link_prediction_experiment',
    'evaluate_link_prediction',
    
    # Graph classification
    'run_graph_classification_experiment',
    'evaluate_graph_classification',
    
    # Ablation study
    'run_ablation_study',
    'ablation_components'
]