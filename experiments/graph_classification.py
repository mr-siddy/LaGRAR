"""
Graph classification experiments for LaGRAR.

This module implements the graph classification experiments described in Section 6
and Table 3 of the paper. It includes functions to run graph classification experiments
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
from sklearn.metrics import accuracy_score, f1_score

from torch_geometric.loader import DataLoader

from .config import (
    GRAPH_CLASSIFICATION_CONFIGS,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_PATIENCE,
    DEFAULT_RUNS,
    DEFAULT_SEEDS,
    DEVICE
)

def run_graph_classification_experiment(
    model_class,
    dataset_name: str,
    dataset,
    split_idx: Dict[str, torch.Tensor],
    model_params: Optional[Dict] = None,
    use_lagrar: bool = True,
    backbone: str = 'gcn',
    batch_size: int = 32,
    epochs: int = DEFAULT_EPOCHS,
    eval_steps: int = DEFAULT_EVAL_STEPS,
    patience: int = DEFAULT_PATIENCE,
    runs: int = DEFAULT_RUNS,
    seeds: List[int] = DEFAULT_SEEDS,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Run graph classification experiment for LaGRAR as described in the paper.
    
    Args:
        model_class: Model class to use (LaGRAR or baseline)
        dataset_name: Name of the dataset
        dataset: PyG dataset
        split_idx: Dictionary with train/val/test indices
        model_params: Dictionary with model parameters
        use_lagrar: Whether to use LaGRAR or baseline model
        backbone: Backbone model ('gcn' or 'gin')
        batch_size: Batch size for training
        epochs: Number of epochs
        eval_steps: Number of steps between evaluations
        patience: Patience for early stopping
        runs: Number of runs
        seeds: Random seeds for runs
        save_dir: Directory to save results
        verbose: Whether to print progress
    
    Returns:
        train_metrics: Dictionary with train metrics (accuracy, f1) for each run
        test_metrics: Dictionary with test metrics (accuracy, f1) for each run
    """
    assert len(seeds) >= runs, f"Not enough seeds for {runs} runs"
    
    # Get model parameters
    if model_params is None:
        # Get default parameters from config
        model_name = f"LaGRAR-{backbone.upper()}" if use_lagrar else f"{backbone.upper()}"
        if dataset_name.upper() in GRAPH_CLASSIFICATION_CONFIGS and model_name in GRAPH_CLASSIFICATION_CONFIGS[dataset_name.upper()]:
            model_params = GRAPH_CLASSIFICATION_CONFIGS[dataset_name.upper()][model_name]
        else:
            # Fallback to default parameters
            model_params = {
                'hidden_dim': 64,
                'num_layers': 3,
                'dropout': 0.5,
                'lr': 0.01,
                'weight_decay': 5e-4
            }
    
    # Prepare results containers
    train_metrics = {
        'accuracy': [],
        'f1_macro': [],
        'f1_micro': [],
        'f1_weighted': []
    }
    test_metrics = {
        'accuracy': [],
        'f1_macro': [],
        'f1_micro': [],
        'f1_weighted': []
    }
    
    # Run multiple times with different seeds
    for run_idx in range(runs):
        seed = seeds[run_idx]
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if verbose:
            print(f"Run {run_idx+1}/{runs} with seed {seed}")
        
        # Initialize model
        model = model_class(
            input_dim=dataset.num_features,
            num_classes=dataset.num_classes,
            backbone=backbone,
            use_lagrar=use_lagrar,
            task='graph_classification',
            **{k: v for k, v in model_params.items() if k not in ['lr', 'weight_decay']}
        ).to(DEVICE)
        
        # Prepare optimizer
        optimizer = Adam(
            model.parameters(),
            lr=model_params.get('lr', 0.01),
            weight_decay=model_params.get('weight_decay', 5e-4)
        )
        
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
        best_epoch = 0
        train_loss_history = []
        val_acc_history = []
        patience_counter = 0
        
        t_start = time.time()
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
            
            avg_loss = total_loss / len(train_loader.dataset)
            train_loss_history.append(avg_loss)
            
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
                val_acc_history.append(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model state
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if verbose and ((epoch + 1) % (eval_steps * 10) == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")
                
                # Early stopping
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - t_start
        if verbose:
            print(f"Training completed in {training_time:.2f}s")
            print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
        
        # Load best model for final evaluation
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
        
        # Final evaluation
        model.eval()
        
        # Train set evaluation
        train_preds = []
        train_labels = []
        
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                train_preds.append(pred.cpu().numpy())
                train_labels.append(batch.y.cpu().numpy())
        
        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)
        
        # Test set evaluation
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                test_preds.append(pred.cpu().numpy())
                test_labels.append(batch.y.cpu().numpy())
        
        test_preds = np.concatenate(test_preds)
        test_labels = np.concatenate(test_labels)
        
        # Compute metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1_macro = f1_score(train_labels, train_preds, average='macro')
        train_f1_micro = f1_score(train_labels, train_preds, average='micro')
        train_f1_weighted = f1_score(train_labels, train_preds, average='weighted')
        
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1_macro = f1_score(test_labels, test_preds, average='macro')
        test_f1_micro = f1_score(test_labels, test_preds, average='micro')
        test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
        
        if verbose:
            print(f"Train accuracy: {train_acc:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")
        
        # Record metrics
        train_metrics['accuracy'].append(train_acc)
        train_metrics['f1_macro'].append(train_f1_macro)
        train_metrics['f1_micro'].append(train_f1_micro)
        train_metrics['f1_weighted'].append(train_f1_weighted)
        
        test_metrics['accuracy'].append(test_acc)
        test_metrics['f1_macro'].append(test_f1_macro)
        test_metrics['f1_micro'].append(test_f1_micro)
        test_metrics['f1_weighted'].append(test_f1_weighted)
    
    # Summarize results
    if verbose:
        print("\nSummary:")
        print(f"Train accuracy: {np.mean(train_metrics['accuracy']):.4f} ± {np.std(train_metrics['accuracy']):.4f}")
        print(f"Test accuracy: {np.mean(test_metrics['accuracy']):.4f} ± {np.std(test_metrics['accuracy']):.4f}")
    
    # Save results
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        results = {
            'dataset': dataset_name,
            'model': f"LaGRAR-{backbone.upper()}" if use_lagrar else f"{backbone.upper()}",
            'train_acc_mean': np.mean(train_metrics['accuracy']),
            'train_acc_std': np.std(train_metrics['accuracy']),
            'test_acc_mean': np.mean(test_metrics['accuracy']),
            'test_acc_std': np.std(test_metrics['accuracy']),
            'train_f1_macro_mean': np.mean(train_metrics['f1_macro']),
            'train_f1_macro_std': np.std(train_metrics['f1_macro']),
            'test_f1_macro_mean': np.mean(test_metrics['f1_macro']),
            'test_f1_macro_std': np.std(test_metrics['f1_macro']),
            'train_f1_micro_mean': np.mean(train_metrics['f1_micro']),
            'train_f1_micro_std': np.std(train_metrics['f1_micro']),
            'test_f1_micro_mean': np.mean(test_metrics['f1_micro']),
            'test_f1_micro_std': np.std(test_metrics['f1_micro']),
            'train_f1_weighted_mean': np.mean(train_metrics['f1_weighted']),
            'train_f1_weighted_std': np.std(train_metrics['f1_weighted']),
            'test_f1_weighted_mean': np.mean(test_metrics['f1_weighted']),
            'test_f1_weighted_std': np.std(test_metrics['f1_weighted']),
        }
        
        pd.DataFrame([results]).to_csv(
            os.path.join(save_dir, f"{dataset_name}_{results['model']}_graph_clf_results.csv"),
            index=False
        )
    
    return train_metrics, test_metrics

def evaluate_graph_classification(
    results_dir: str,
    dataset_names: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate graph classification results as presented in Table 3 of the paper.
    
    Args:
        results_dir: Directory with saved results
        dataset_names: List of dataset names to include
        models: List of models to include
        metrics: List of metrics to include
        output_file: File to save aggregated results
        verbose: Whether to print results
    
    Returns:
        DataFrame with aggregated results
    """
    if dataset_names is None:
        dataset_names = ['REDDIT-BINARY', 'IMDB-BINARY', 'COLLAB', 'MUTAG', 'PROTEINS', 'ENZYMES']
    
    if models is None:
        models = [
            'GCN', 'K-NN', 'MinCutPool', 'GCN DIGL', 'SDRF', 'FoSR', 'BORF', 'LaGRAR',
            'GIN', 'K-NN', 'MinCutPool', 'GIN DIGL', 'SDRF', 'FoSR', 'BORF', 'LaGRAR'
        ]
    
    if metrics is None:
        metrics = ['accuracy']
    
    # Collect all results files
    all_files = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith('_graph_clf_results.csv'):
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
    
    # Filter by dataset names and models
    if dataset_names:
        results_df = results_df[results_df['dataset'].isin([d.upper() for d in dataset_names])]
    
    if models:
        results_df = results_df[results_df['model'].str.contains('|'.join(models), case=False)]
    
    # Create a table for each backbone type (GCN and GIN)
    table_data = []
    
    # Group by backbone first
    backbones = ['GCN', 'GIN']
    
    for backbone in backbones:
        # Filter by backbone
        backbone_results = results_df[results_df['model'].str.contains(backbone, case=False)]
        
        if backbone_results.empty:
            continue
        
        for dataset in sorted(backbone_results['dataset'].unique()):
            # This will store rows for the current dataset
            dataset_rows = []
            
            for model in models:
                if backbone.lower() not in model.lower():
                    continue
                    
                model_results = backbone_results[
                    (backbone_results['dataset'] == dataset) & 
                    (backbone_results['model'].str.contains(model, case=False))
                ]
                
                if model_results.empty:
                    continue
                
                row_data = {
                    'backbone': backbone,
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
                        row_data[metric] = f"{mean_val:.2f} ± {std_val:.2f}"
                        row_data[f"{metric}_mean"] = mean_val
                        row_data[f"{metric}_std"] = std_val
                
                dataset_rows.append(row_data)
            
            # Sort rows by accuracy within this dataset
            if dataset_rows:
                dataset_rows_sorted = sorted(
                    dataset_rows, 
                    key=lambda x: x.get(f"{metrics[0]}_mean", 0), 
                    reverse=True
                )
                table_data.extend(dataset_rows_sorted)
    
    # Create table
    table_df = pd.DataFrame(table_data)
    
    # Print results in the format of Table 3
    if verbose:
        headers = ['Backbone', 'Dataset', 'Model'] + metrics
        
        print(f"{'='*100}")
        print(f"{'Graph Classification Results':^100}")
        print(f"{'='*100}")
        
        for backbone in backbones:
            backbone_data = table_df[table_df['backbone'] == backbone]
            
            if backbone_data.empty:
                continue
                
            print(f"\n{backbone} Backbone:")
            print("-" * 100)
            
            # Group by dataset
            for dataset in sorted(backbone_data['dataset'].unique()):
                dataset_results = backbone_data[backbone_data['dataset'] == dataset]
                
                if dataset_results.empty:
                    continue
                    
                print(f"\nDataset: {dataset}")
                
                # Format the table
                header_fmt = "| {:<20} |" + " {:<15} |" * len(metrics)
                print("-" * (24 + 17 * len(metrics)))
                print(header_fmt.format('Model', *metrics))
                print("-" * (24 + 17 * len(metrics)))
                
                # Sort by the first metric
                dataset_results = dataset_results.sort_values(
                    by=f"{metrics[0]}_mean", 
                    ascending=False
                ).reset_index(drop=True)
                
                best_model = dataset_results.iloc[0]['model']
                
                for idx, row in dataset_results.iterrows():
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
        
        table_df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"\nResults saved to {output_file}")
    
    return table_df