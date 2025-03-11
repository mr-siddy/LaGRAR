"""
Ablation study experiments for LaGRAR.

This module implements the ablation studies described in Section 6 and Tables 4, 9, 10
of the paper. It includes functions to run experiments with different components of
LaGRAR removed to analyze their importance.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from torch.optim import Adam
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import (
    ABLATION_COMPONENTS,
    NODE_CLASSIFICATION_CONFIGS,
    LINK_PREDICTION_CONFIGS,
    GRAPH_CLASSIFICATION_CONFIGS,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_RUNS,
    DEFAULT_SEEDS,
    DEVICE
)

# Define ablation components
def ablation_components(use_task_loss: bool = True, 
                        use_positive_loss: bool = True, 
                        use_negative_loss: bool = True,
                        use_locality_preserving: bool = True) -> Dict[str, bool]:
    """
    Create ablation component configuration.
    
    Args:
        use_task_loss: Whether to use task-specific loss
        use_positive_loss: Whether to use positive substructure loss (Lpos)
        use_negative_loss: Whether to use negative adversarial loss (Lneg)
        use_locality_preserving: Whether to use locality preserving sparsification (G3i(λ))
    
    Returns:
        Dictionary with ablation component configuration
    """
    return {
        'use_task_loss': use_task_loss,
        'use_positive_loss': use_positive_loss, 
        'use_negative_loss': use_negative_loss,
        'use_locality_preserving': use_locality_preserving
    }

def get_ablation_name(components: Dict[str, bool]) -> str:
    """
    Get name for ablation configuration.
    
    Args:
        components: Dictionary with ablation component configuration
    
    Returns:
        Ablation name
    """
    if not components['use_task_loss']:
        return 'w/o Ltask'
    elif not components['use_positive_loss']:
        return 'w/o Lpos'
    elif not components['use_negative_loss']:
        return 'w/o Lneg'
    elif not components['use_locality_preserving']:
        return 'w/o G3(λ)'
    else:
        return 'LaGRAR-full'

def run_ablation_study(
    model_class,
    dataset_name: str,
    dataset,
    task: str,
    split_idx: Dict[str, torch.Tensor],
    model_params: Optional[Dict] = None,
    backbone: str = 'gcn',
    batch_size: int = 32,
    epochs: int = DEFAULT_EPOCHS,
    eval_steps: int = DEFAULT_EVAL_STEPS,
    patience: int = DEFAULT_PATIENCE,
    runs: int = DEFAULT_RUNS,
    seeds: List[int] = DEFAULT_SEEDS,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation study for LaGRAR as described in the paper.
    
    Args:
        model_class: LaGRAR model class
        dataset_name: Name of the dataset
        dataset: PyG dataset
        task: Task type ('node_classification', 'link_prediction', 'graph_classification')
        split_idx: Dictionary with split indices
        model_params: Dictionary with model parameters
        backbone: Backbone model ('gcn' or 'gat' for node tasks, 'gcn' or 'gin' for graph tasks)
        batch_size: Batch size for training (only for graph classification)
        epochs: Number of epochs
        eval_steps: Number of steps between evaluations
        patience: Patience for early stopping
        runs: Number of runs
        seeds: Random seeds for runs
        save_dir: Directory to save results
        verbose: Whether to print progress
    
    Returns:
        Dictionary with ablation results for each component
    """
    assert task in ['node_classification', 'link_prediction', 'graph_classification'], \
        f"Task {task} not supported. Use 'node_classification', 'link_prediction', or 'graph_classification'."
    
    # Get model parameters based on task
    if model_params is None:
        model_name = f"LaGRAR-{backbone.upper()}"
        
        if task == 'node_classification':
            config_dict = NODE_CLASSIFICATION_CONFIGS
        elif task == 'link_prediction':
            config_dict = LINK_PREDICTION_CONFIGS
        else:  # graph_classification
            config_dict = GRAPH_CLASSIFICATION_CONFIGS
            dataset_name = dataset_name.upper()  # Graph classification datasets are uppercase
        
        if dataset_name in config_dict and model_name in config_dict[dataset_name]:
            model_params = config_dict[dataset_name][model_name]
        else:
            # Fallback to default parameters
            model_params = {
                'hidden_dim': 64,
                'num_layers': 2 if task in ['node_classification', 'link_prediction'] else 3,
                'dropout': 0.5,
                'lr': 0.01,
                'weight_decay': 5e-4
            }
    
    # Define ablation configurations to test
    ablation_configs = [
        ablation_components(use_task_loss=False, use_positive_loss=True, use_negative_loss=True, use_locality_preserving=True),
        ablation_components(use_task_loss=True, use_positive_loss=False, use_negative_loss=True, use_locality_preserving=True),
        ablation_components(use_task_loss=True, use_positive_loss=True, use_negative_loss=False, use_locality_preserving=True),
        ablation_components(use_task_loss=True, use_positive_loss=True, use_negative_loss=True, use_locality_preserving=False)
    ]
    
    # Add full model configuration
    ablation_configs.append(
        ablation_components(use_task_loss=True, use_positive_loss=True, use_negative_loss=True, use_locality_preserving=True)
    )
    
    # Store results for each ablation configuration
    ablation_results = {}
    
    # Run experiment for each ablation configuration
    for config in ablation_configs:
        ablation_name = get_ablation_name(config)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running ablation: {ablation_name}")
            print(f"{'='*80}")
        
        # Run experiments with this configuration
        test_metrics = []
        
        for run_idx in range(runs):
            seed = seeds[run_idx]
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            if verbose:
                print(f"Run {run_idx+1}/{runs} with seed {seed}")
            
            # Initialize model with ablation configuration
            model_args = {
                'input_dim': dataset.num_features,
                'num_classes': dataset.num_classes,
                'backbone': backbone,
                'use_lagrar': True,
                'task': task,
                **{k: v for k, v in model_params.items() if k not in ['lr', 'weight_decay']},
                **config  # Add ablation configuration
            }
            
            model = model_class(**model_args).to(DEVICE)
            
            # Prepare optimizer
            optimizer = Adam(
                model.parameters(),
                lr=model_params.get('lr', 0.01),
                weight_decay=model_params.get('weight_decay', 5e-4)
            )
            
            # Training and evaluation depend on the task
            if task == 'node_classification':
                # Run node classification
                result = run_node_classification_ablation(
                    model, optimizer, dataset, split_idx, 
                    epochs, eval_steps, patience, verbose
                )
            elif task == 'link_prediction':
                # Run link prediction
                result = run_link_prediction_ablation(
                    model, optimizer, dataset, split_idx, 
                    epochs, eval_steps, patience, verbose
                )
            else:  # graph_classification
                # Run graph classification
                result = run_graph_classification_ablation(
                    model, optimizer, dataset, split_idx, batch_size,
                    epochs, eval_steps, patience, verbose
                )
            
            test_metrics.append(result)
        
        # Calculate mean and standard deviation
        test_mean = np.mean(test_metrics)
        test_std = np.std(test_metrics)
        
        ablation_results[ablation_name] = {
            'mean': test_mean,
            'std': test_std
        }
        
        if verbose:
            print(f"\n{ablation_name} - Test {get_primary_metric(task)}: {test_mean:.4f} ± {test_std:.4f}")
    
    # Save results
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # Create results dataframe
        results_data = []
        
        for ablation_name, metrics in ablation_results.items():
            results_data.append({
                'dataset': dataset_name,
                'task': task,
                'backbone': backbone,
                'ablation': ablation_name,
                'metric_mean': metrics['mean'],
                'metric_std': metrics['std']
            })
        
        pd.DataFrame(results_data).to_csv(
            os.path.join(save_dir, f"{dataset_name}_{backbone}_ablation_results.csv"),
            index=False
        )
    
    return ablation_results

# Helper functions for different tasks
def get_primary_metric(task: str) -> str:
    """
    Get primary metric for task.
    
    Args:
        task: Task type
    
    Returns:
        Primary metric name
    """
    if task == 'node_classification':
        return 'accuracy'
    elif task == 'link_prediction':
        return 'ROC-AUC'
    else:  # graph_classification
        return 'accuracy'

def run_node_classification_ablation(
    model, 
    optimizer, 
    dataset, 
    split_idx, 
    epochs, 
    eval_steps, 
    patience, 
    verbose
) -> float:
    """
    Run node classification ablation.
    
    Args:
        model: Model with ablation configuration
        optimizer: Optimizer
        dataset: Dataset
        split_idx: Split indices
        epochs: Number of epochs
        eval_steps: Evaluation steps
        patience: Early stopping patience
        verbose: Whether to print progress
    
    Returns:
        Test accuracy
    """
    # Get split indices
    train_idx = split_idx['train_idx'].to(DEVICE)
    val_idx = split_idx['val_idx'].to(DEVICE)
    test_idx = split_idx['test_idx'].to(DEVICE)
    
    # Get data
    data = dataset[0].to(DEVICE)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if (epoch + 1) % eval_steps == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # Validation accuracy
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                val_acc = accuracy_score(
                    data.y[val_idx].cpu().numpy(),
                    pred[val_idx].cpu().numpy()
                )
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    
                    # Save best model state
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    break
    
    # Load best model for test evaluation
    if best_model_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = accuracy_score(
            data.y[test_idx].cpu().numpy(),
            pred[test_idx].cpu().numpy()
        )
    
    return test_acc

def run_link_prediction_ablation(
    model, 
    optimizer, 
    dataset, 
    split_idx, 
    epochs, 
    eval_steps, 
    patience, 
    verbose
) -> float:
    """
    Run link prediction ablation.
    
    Args:
        model: Model with ablation configuration
        optimizer: Optimizer
        dataset: Dataset
        split_idx: Split indices
        epochs: Number of epochs
        eval_steps: Evaluation steps
        patience: Early stopping patience
        verbose: Whether to print progress
    
    Returns:
        Test ROC-AUC
    """
    # Get split indices
    train_pos = split_idx['train_pos'].to(DEVICE)
    train_neg = split_idx['train_neg'].to(DEVICE)
    val_pos = split_idx['val_pos'].to(DEVICE)
    val_neg = split_idx['val_neg'].to(DEVICE)
    test_pos = split_idx['test_pos'].to(DEVICE)
    test_neg = split_idx['test_neg'].to(DEVICE)
    
    # Get data
    data = dataset[0].to(DEVICE)
    
    # Training loop
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass - get embeddings
        z = model.encode(data.x, data.edge_index)
        
        # Compute edge predictions for positive and negative samples
        edge_pos_pred = model.decode_edges(z, train_pos)
        edge_neg_pred = model.decode_edges(z, train_neg)
        
        # Prepare labels: 1 for positive edges, 0 for negative edges
        edge_pos_labels = torch.ones(train_pos.size(1), device=DEVICE)
        edge_neg_labels = torch.zeros(train_neg.size(1), device=DEVICE)
        
        # Combine predictions and labels
        edge_pred = torch.cat([edge_pos_pred, edge_neg_pred], dim=0)
        edge_labels = torch.cat([edge_pos_labels, edge_neg_labels], dim=0)
        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(edge_pred, edge_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if (epoch + 1) % eval_steps == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # Get embeddings
                z = model.encode(data.x, data.edge_index)
                
                # Validation predictions
                val_pos_pred = model.decode_edges(z, val_pos)
                val_neg_pred = model.decode_edges(z, val_neg)
                
                # Prepare labels
                val_pos_labels = torch.ones(val_pos.size(1))
                val_neg_labels = torch.zeros(val_neg.size(1))
                
                # Combine predictions and labels
                val_pred = torch.cat([val_pos_pred, val_neg_pred], dim=0).cpu().numpy()
                val_labels = torch.cat([val_pos_labels, val_neg_labels], dim=0).cpu().numpy()
                
                # Compute validation ROC-AUC
                val_auc = roc_auc_score(val_labels, val_pred)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    
                    # Save best model state
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    break
    
    # Load best model for test evaluation
    if best_model_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        # Get embeddings
        z = model.encode(data.x, data.edge_index)
        
        # Test predictions
        test_pos_pred = model.decode_edges(z, test_pos).cpu().numpy()
        test_neg_pred = model.decode_edges(z, test_neg).cpu().numpy()
        
        # Prepare labels
        test_pos_labels = np.ones(test_pos.size(1))
        test_neg_labels = np.zeros(test_neg.size(1))
        
        # Combine predictions and labels
        test_pred = np.concatenate([test_pos_pred, test_neg_pred])
        test_labels = np.concatenate([test_pos_labels, test_neg_labels])
        
        # Compute test ROC-AUC
        test_auc = roc_auc_score(test_labels, test_pred)
    
    return test_auc

def run_graph_classification_ablation(
    model, 
    optimizer, 
    dataset, 
    split_idx, 
    batch_size,
    epochs, 
    eval_steps, 
    patience, 
    verbose
) -> float:
    """
    Run graph classification ablation.
    
    Args:
        model: Model with ablation configuration
        optimizer: Optimizer
        dataset: Dataset
        split_idx: Split indices
        batch_size: Batch size
        epochs: Number of epochs
        eval_steps: Evaluation steps
        patience: Early stopping patience
        verbose: Whether to print progress
    
    Returns:
        Test accuracy
    """
    from torch_geometric.loader import DataLoader
    
    # Create data loaders
    train_idx = split_idx['train_idx']
    val_idx = split_idx['val_idx']
    test_idx = split_idx['test_idx']
    
    train_dataset = dataset[train_idx.tolist()]
    val_dataset = dataset[val_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
        
        # Evaluation
        if (epoch + 1) % eval_steps == 0 or epoch == epochs - 1:
            model.eval()
            
            # Validation accuracy
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    pred = out.argmax(dim=1)
                    val_correct += (pred == batch.y).sum().item()
                    val_total += batch.num_graphs
            
            val_acc = val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model state
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                break
    
    # Load best model for test evaluation
    if best_model_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
    
    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            test_correct += (pred == batch.y).sum().item()
            test_total += batch.num_graphs
    
    test_acc = test_correct / test_total
    
    return test_acc