"""
Link prediction experiments for LaGRAR.

This module implements the link prediction experiments described in Section 6
and Table 2 of the paper. It includes functions to run link prediction experiments
and evaluate the performance of LaGRAR against baseline models.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from torch.optim import Adam
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import (
    LINK_PREDICTION_CONFIGS,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_RUNS,
    DEFAULT_SEEDS,
    DEVICE
)

def run_link_prediction_experiment(
    model_class,
    dataset_name: str,
    dataset,
    split_idx: Dict[str, torch.Tensor],
    model_params: Optional[Dict] = None,
    use_lagrar: bool = True,
    backbone: str = 'gcn',
    epochs: int = DEFAULT_EPOCHS,
    eval_steps: int = DEFAULT_EVAL_STEPS,
    patience: int = DEFAULT_PATIENCE,
    runs: int = DEFAULT_RUNS,
    seeds: List[int] = DEFAULT_SEEDS,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Run link prediction experiment for LaGRAR as described in the paper.
    
    Args:
        model_class: Model class to use (LaGRAR or baseline)
        dataset_name: Name of the dataset
        dataset: PyG dataset
        split_idx: Dictionary with train/val/test split indices
        model_params: Dictionary with model parameters
        use_lagrar: Whether to use LaGRAR or baseline model
        backbone: Backbone model ('gcn' or 'gat')
        epochs: Number of epochs
        eval_steps: Number of steps between evaluations
        patience: Patience for early stopping
        runs: Number of runs
        seeds: Random seeds for runs
        save_dir: Directory to save results
        verbose: Whether to print progress
    
    Returns:
        train_metrics: Dictionary with train metrics (ROC-AUC, AP) for each run
        test_metrics: Dictionary with test metrics (ROC-AUC, AP) for each run
    """
    assert len(seeds) >= runs, f"Not enough seeds for {runs} runs"
    
    # Get model parameters
    if model_params is None:
        # Get default parameters from config
        model_name = f"LaGRAR-{backbone.upper()}" if use_lagrar else f"{backbone.upper()}"
        if dataset_name.lower() in LINK_PREDICTION_CONFIGS and model_name in LINK_PREDICTION_CONFIGS[dataset_name.lower()]:
            model_params = LINK_PREDICTION_CONFIGS[dataset_name.lower()][model_name]
        else:
            # Fallback to default parameters
            model_params = {
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.5,
                'lr': 0.01,
                'weight_decay': 5e-4
            }
    
    # Prepare results containers
    train_metrics = {
        'roc_auc': [],
        'ap': []
    }
    test_metrics = {
        'roc_auc': [],
        'ap': []
    }
    
    # Run multiple times with different seeds
    for run_idx in range(runs):
        seed = seeds[run_idx]
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if verbose:
            print(f"Run {run_idx+1}/{runs} with seed {seed}")
        
        # Initialize model
        data = dataset[0]  # Get the first graph from the dataset
        model = model_class(
            input_dim=dataset.num_features,
            num_classes=2,  # Binary classification for link prediction
            backbone=backbone,
            use_lagrar=use_lagrar,
            task='link_prediction',
            **{k: v for k, v in model_params.items() if k not in ['lr', 'weight_decay']}
        ).to(DEVICE)
        
        # Prepare optimizer
        optimizer = Adam(
            model.parameters(),
            lr=model_params.get('lr', 0.01),
            weight_decay=model_params.get('weight_decay', 5e-4)
        )
        
        # Get split indices
        train_pos = split_idx['train_pos'].to(DEVICE)
        train_neg = split_idx['train_neg'].to(DEVICE)
        val_pos = split_idx['val_pos'].to(DEVICE)
        val_neg = split_idx['val_neg'].to(DEVICE)
        test_pos = split_idx['test_pos'].to(DEVICE)
        test_neg = split_idx['test_neg'].to(DEVICE)
        
        # Move data to device
        data = data.to(DEVICE)
        
        # Training loop
        best_val_auc = 0
        best_epoch = 0
        train_loss_history = []
        val_auc_history = []
        patience_counter = 0
        
        t_start = time.time()
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
            
            train_loss_history.append(loss.item())
            
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
                    val_auc_history.append(val_auc)
                    
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_epoch = epoch
                        patience_counter = 0
                        
                        # Save best model state
                        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                    
                    if verbose and ((epoch + 1) % (eval_steps * 10) == 0 or epoch == epochs - 1):
                        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {loss.item():.4f}, Val AUC = {val_auc:.4f}")
                    
                    # Early stopping
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
        
        training_time = time.time() - t_start
        if verbose:
            print(f"Training completed in {training_time:.2f}s")
            print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
        
        # Load best model for final evaluation
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            # Get embeddings
            z = model.encode(data.x, data.edge_index)
            
            # Training set predictions
            train_pos_pred = model.decode_edges(z, train_pos).cpu().numpy()
            train_neg_pred = model.decode_edges(z, train_neg).cpu().numpy()
            
            # Test set predictions
            test_pos_pred = model.decode_edges(z, test_pos).cpu().numpy()
            test_neg_pred = model.decode_edges(z, test_neg).cpu().numpy()
            
            # Prepare labels
            train_pos_labels = np.ones(train_pos.size(1))
            train_neg_labels = np.zeros(train_neg.size(1))
            test_pos_labels = np.ones(test_pos.size(1))
            test_neg_labels = np.zeros(test_neg.size(1))
            
            # Combine predictions and labels
            train_pred = np.concatenate([train_pos_pred, train_neg_pred])
            train_labels = np.concatenate([train_pos_labels, train_neg_labels])
            test_pred = np.concatenate([test_pos_pred, test_neg_pred])
            test_labels = np.concatenate([test_pos_labels, test_neg_labels])
            
            # Compute metrics
            train_roc_auc = roc_auc_score(train_labels, train_pred)
            train_ap = average_precision_score(train_labels, train_pred)
            test_roc_auc = roc_auc_score(test_labels, test_pred)
            test_ap = average_precision_score(test_labels, test_pred)
            
            if verbose:
                print(f"Train ROC-AUC: {train_roc_auc:.4f}, AP: {train_ap:.4f}")
                print(f"Test ROC-AUC: {test_roc_auc:.4f}, AP: {test_ap:.4f}")
        
        # Record metrics
        train_metrics['roc_auc'].append(train_roc_auc)
        train_metrics['ap'].append(train_ap)
        test_metrics['roc_auc'].append(test_roc_auc)
        test_metrics['ap'].append(test_ap)
    
    # Summarize results
    if verbose:
        print("\nSummary:")
        print(f"Train ROC-AUC: {np.mean(train_metrics['roc_auc']):.4f} ± {np.std(train_metrics['roc_auc']):.4f}")
        print(f"Test ROC-AUC: {np.mean(test_metrics['roc_auc']):.4f} ± {np.std(test_metrics['roc_auc']):.4f}")
    
    # Save results
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        results = {
            'dataset': dataset_name,
            'model': f"LaGRAR-{backbone.upper()}" if use_lagrar else f"{backbone.upper()}",
            'train_roc_auc_mean': np.mean(train_metrics['roc_auc']),
            'train_roc_auc_std': np.std(train_metrics['roc_auc']),
            'test_roc_auc_mean': np.mean(test_metrics['roc_auc']),
            'test_roc_auc_std': np.std(test_metrics['roc_auc']),
            'train_ap_mean': np.mean(train_metrics['ap']),
            'train_ap_std': np.std(train_metrics['ap']),
            'test_ap_mean': np.mean(test_metrics['ap']),
            'test_ap_std': np.std(test_metrics['ap']),
        }
        
        pd.DataFrame([results]).to_csv(
            os.path.join(save_dir, f"{dataset_name}_{results['model']}_link_pred_results.csv"),
            index=False
        )
    
    return train_metrics, test_metrics

def evaluate_link_prediction(
    results_dir: str,
    dataset_groups: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate link prediction results as presented in Table 2 of the paper.
    
    Args:
        results_dir: Directory with saved results
        dataset_groups: List of dataset groups to include ('SHD', 'MHD', 'MHOD')
        models: List of models to include
        metrics: List of metrics to include
        output_file: File to save aggregated results
        verbose: Whether to print results
    
    Returns:
        DataFrame with aggregated results
    """
    if dataset_groups is None:
        dataset_groups = ['SHD', 'MHD', 'MHOD']
    
    if models is None:
        models = [
            'GCN', 'GAT', 'GAT-v2', 'MixHop', 'TransformerConv',
            'SDRF', 'FoSR', 'BORF', 'DIGL', 'AFR-3',
            'HGCN-Poincare', 'HGCN-Hyperboloid',
            'LGR-GCN', 'LGR-GAT', 'LaGRAR-GCN', 'LaGRAR-GAT'
        ]
    
    if metrics is None:
        metrics = ['roc_auc']
    
    # Map dataset groups to datasets
    dataset_mapping = {
        'SHD': ['cornell', 'wisconsin', 'texas'],
        'MHD': ['chameleon', 'squirrel', 'actor'],
        'MHOD': ['cora', 'citeseer', 'pubmed']
    }
    
    # Collect all results files
    all_files = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith('_link_pred_results.csv'):
                all_files.append(os.path.join(root, file))
    
    # Load results
    all_results = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_results:
        print("No results found!")
        return pd.DataFrame()
    
    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Filter by dataset groups and models
    datasets = []
    for group in dataset_groups:
        datasets.extend(dataset_mapping.get(group, []))
    
    if datasets:
        results_df = results_df[results_df['dataset'].isin(datasets)]
    
    if models:
        results_df = results_df[results_df['model'].isin(models)]
    
    # Create a table for each dataset and model combination
    table_data = []
    
    for dataset in sorted(results_df['dataset'].unique()):
        dataset_group = next((group for group, ds_list in dataset_mapping.items() if dataset in ds_list), 'Other')
        
        for model in models:
            model_results = results_df[(results_df['dataset'] == dataset) & (results_df['model'] == model)]
            
            if model_results.empty:
                continue
            
            row_data = {
                'dataset_group': dataset_group,
                'dataset': dataset,
                'model': model
            }
            
            # Add metrics
            for metric in metrics:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"
                
                if mean_col in model_results.columns and std_col in model_results.columns:
                    mean_val = model_results[mean_col].values[0] * 100  # Convert to percentage
                    std_val = model_results[std_col].values[0] * 100    # Convert to percentage
                    row_data[metric] = f"{mean_val:.2f}±{std_val:.2f}"
                    row_data[f"{metric}_mean"] = mean_val
                    row_data[f"{metric}_std"] = std_val
            
            table_data.append(row_data)
    
    # Create table
    table_df = pd.DataFrame(table_data)
    
    # Group by dataset and sort by metric
    grouped_tables = {}
    
    for dataset in sorted(table_df['dataset'].unique()):
        dataset_results = table_df[table_df['dataset'] == dataset].sort_values(
            by=f"{metrics[0]}_mean", ascending=False
        ).reset_index(drop=True)
        
        grouped_tables[dataset] = dataset_results
    
    # Combine all tables
    final_table = pd.concat(grouped_tables.values(), ignore_index=True)
    
    # Print results in the format of Table 2
    if verbose:
        headers = ['Dataset', 'Model'] + metrics
        
        print(f"{'='*80}")
        print(f"{'Link Prediction Results':^80}")
        print(f"{'='*80}")
        
        for group_name, group_datasets in dataset_mapping.items():
            if not any(ds in final_table['dataset'].unique() for ds in group_datasets):
                continue
                
            print(f"\n{group_name} Datasets:")
            print("-" * 80)
            
            for dataset in group_datasets:
                if dataset not in final_table['dataset'].unique():
                    continue
                    
                print(f"\nDataset: {dataset}")
                
                dataset_results = final_table[final_table['dataset'] == dataset]
                best_model = dataset_results.iloc[0]['model']
                best_value = dataset_results.iloc[0][f"{metrics[0]}_mean"]
                
                # Format the table
                header_fmt = "| {:<20} |" + " {:<15} |" * len(metrics)
                print("-" * (24 + 17 * len(metrics)))
                print(header_fmt.format('Model', *metrics))
                print("-" * (24 + 17 * len(metrics)))
                
                for _, row in dataset_results.iterrows():
                    model = row['model']
                    values = [row[metric] for metric in metrics]
                    
                    is_best = (model == best_model)
                    
                    row_fmt = "| {}{:<19} |" + " {:<15} |" * len(metrics)
                    print(row_fmt.format('*' if is_best else ' ', model, *values))
                
                print("-" * (24 + 17 * len(metrics)))
    
    # Save results if output file is provided
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        final_table.to_csv(output_file, index=False)
        
        if verbose:
            print(f"\nResults saved to {output_file}")
    
    return final_table