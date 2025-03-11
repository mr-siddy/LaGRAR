"""
Configuration parameters for LaGRAR experiments.

This module defines the hyperparameters and configurations for all experiments
as reported in Section 5 and Appendix A.8 of the paper.
"""

import torch

# Dataset groups as in the paper (Section 5)
SHD_DATASETS = ['cornell', 'wisconsin', 'texas']  # Small heterophilic datasets
MHD_DATASETS = ['chameleon', 'squirrel', 'actor']  # Large heterophilic datasets
MHOD_DATASETS = ['cora', 'citeseer', 'pubmed']  # Homophilic datasets
GC_DATASETS = ['REDDIT-BINARY', 'IMDB-BINARY', 'COLLAB', 'MUTAG', 'PROTEINS', 'ENZYMES']  # Graph classification datasets

# Default experiment parameters
DEFAULT_EPOCHS = 400
DEFAULT_EVAL_STEPS = 10
DEFAULT_PATIENCE = 50
DEFAULT_RUNS = 10
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# LaGRAR model hyperparameters
DEFAULT_LAGRAR_PARAMS = {
    'hidden_dim': 64,
    'latent_dim': 64,
    'num_layers': 2,
    'dropout': 0.5,
    'csrf_iterations': 1,
    'alpha': 0.5,
    'beta': 0.5,
    'rho': 0.1,  # Incremental weight for different substructures (Eq. 7)
}

# Optimizer settings
DEFAULT_OPTIMIZER_PARAMS = {
    'lr': 0.01,
    'weight_decay': 5e-4,
}

# Node classification hyperparameters from Table 11 in the paper
NODE_CLASSIFICATION_CONFIGS = {
    # Small heterophilic datasets
    'cornell': {
        'LGR-GCN': {'lr': 0.1, 'weight_decay': 5e-3, 'dropout': 0.2, 'hidden_dim': 64, 'num_layers': 2},
        'LGR-GAT': {'lr': 0.1, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 32, 'num_layers': 1, 'heads': 2},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 1e-3, 'dropout': 0.4, 'hidden_dim': 64, 'num_layers': 1},
        'LaGRAR-GAT': {'lr': 0.1, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 2},
    },
    'wisconsin': {
        'LGR-GCN': {'lr': 0.01, 'weight_decay': 1e-3, 'dropout': 0.6, 'hidden_dim': 64, 'num_layers': 2},
        'LGR-GAT': {'lr': 0.1, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 32, 'num_layers': 1, 'heads': 2},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 1e-3, 'dropout': 0.4, 'hidden_dim': 64, 'num_layers': 1},
        'LaGRAR-GAT': {'lr': 0.1, 'weight_decay': 1e-3, 'dropout': 0.6, 'hidden_dim': 64, 'num_layers': 2},
    },
    'texas': {
        'LGR-GCN': {'lr': 0.05, 'weight_decay': 1e-2, 'dropout': 0.5, 'hidden_dim': 8, 'num_layers': 1},
        'LGR-GAT': {'lr': 0.1, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 8, 'num_layers': 1, 'heads': 8},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 1e-4, 'dropout': 0.4, 'hidden_dim': 64, 'num_layers': 1},
        'LaGRAR-GAT': {'lr': 0.1, 'weight_decay': 1e-3, 'dropout': 0.6, 'hidden_dim': 64, 'num_layers': 2},
    },
    
    # Large heterophilic datasets
    'chameleon': {
        'LGR-GCN': {'lr': 0.01, 'weight_decay': 1e-5, 'dropout': 0.9, 'hidden_dim': 8, 'num_layers': 1},
        'LGR-GAT': {'lr': 0.001, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 8, 'num_layers': 1, 'heads': 8},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.4, 'hidden_dim': 64, 'num_layers': 1},
        'LaGRAR-GAT': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.6, 'hidden_dim': 64, 'num_layers': 2},
    },
    'squirrel': {
        'LGR-GCN': {'lr': 0.01, 'weight_decay': 5e-5, 'dropout': 0.3, 'hidden_dim': 64, 'num_layers': 2},
        'LGR-GAT': {'lr': 0.001, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 32, 'num_layers': 1, 'heads': 2},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 1e-4, 'dropout': 0.4, 'hidden_dim': 64, 'num_layers': 1},
        'LaGRAR-GAT': {'lr': 0.001, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 2},
    },
    'actor': {
        'LGR-GCN': {'lr': 0.01, 'weight_decay': 5e-5, 'dropout': 0.3, 'hidden_dim': 64, 'num_layers': 2},
        'LGR-GAT': {'lr': 0.001, 'weight_decay': 1e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 1, 'heads': 2},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.4, 'hidden_dim': 64, 'num_layers': 1},
        'LaGRAR-GAT': {'lr': 0.1, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 2},
    },
    
    # Homophilic datasets
    'cora': {
        'LGR-GCN': {'lr': 0.01, 'weight_decay': 5e-5, 'dropout': 0.3, 'hidden_dim': 64, 'num_layers': 2},
        'LGR-GAT': {'lr': 0.001, 'weight_decay': 1e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 1, 'heads': 4},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.4, 'hidden_dim': 64, 'num_layers': 1},
        'LaGRAR-GAT': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 2},
    },
    'citeseer': {
        'LGR-GCN': {'lr': 0.1, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 1},
        'LGR-GAT': {'lr': 0.001, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 1, 'heads': 8},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.4, 'hidden_dim': 64, 'num_layers': 1},
        'LaGRAR-GAT': {'lr': 0.001, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 2},
    },
    'pubmed': {
        'LGR-GCN': {'lr': 0.1, 'weight_decay': 5e-5, 'dropout': 0.3, 'hidden_dim': 64, 'num_layers': 2},
        'LGR-GAT': {'lr': 0.1, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 32, 'num_layers': 1, 'heads': 4},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 1e-6, 'dropout': 0.4, 'hidden_dim': 64, 'num_layers': 1},
        'LaGRAR-GAT': {'lr': 0.1, 'weight_decay': 1e-3, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 2},
    },
}

# Link prediction hyperparameters based on those from node classification but adjusted for the link prediction task
LINK_PREDICTION_CONFIGS = {}
for dataset, configs in NODE_CLASSIFICATION_CONFIGS.items():
    LINK_PREDICTION_CONFIGS[dataset] = {}
    for model, config in configs.items():
        # Copy the config and adjust as needed for link prediction
        lp_config = config.copy()
        # Increase dropout for link prediction to reduce overfitting
        lp_config['dropout'] = min(0.7, config.get('dropout', 0.5) + 0.1)
        LINK_PREDICTION_CONFIGS[dataset][model] = lp_config

# Graph classification hyperparameters 
GRAPH_CLASSIFICATION_CONFIGS = {
    'REDDIT-BINARY': {
        'GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
    },
    'IMDB-BINARY': {
        'GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
    },
    'COLLAB': {
        'GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
    },
    'MUTAG': {
        'GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
    },
    'PROTEINS': {
        'GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
    },
    'ENZYMES': {
        'GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GCN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
        'LaGRAR-GIN': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'hidden_dim': 64, 'num_layers': 3},
    },
}

# Ablation study components from Table 4
ABLATION_COMPONENTS = [
    'w/o Ltask',  # Without task-specific loss
    'w/o Lpos',   # Without positive substructure loss
    'w/o Lneg',   # Without negative adversarial loss
    'w/o G3(λ)'   # Without locality preserving sparsification
]

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')