#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for LaGRAR: Task-aware Latent Graph Rewiring.
This script provides command-line functionality to run different experiments 
with the LaGRAR framework.
"""

import os
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.datasets import load_dataset
from data.taskgraph import generate_synthetic_dataset
from models.lagrar import LaGRAR
from models.base import GCN, GAT
from utils.training import train
from utils.evaluation import (
    evaluate_node_classification, 
    evaluate_link_prediction, 
    evaluate_graph_classification,
    evaluate_curvature_distribution,
    evaluate_lagrar_components
)
from utils.visualization import (
    visualize_embeddings,
    visualize_curvature_distribution,
    visualize_loss_components,
    visualize_graph_rewiring
)
from experiments.config import get_config
from experiments.node_classification import run_node_classification_experiment
from experiments.link_prediction import run_link_prediction_experiment
from experiments.graph_classification import run_graph_classification_experiment
from experiments.ablation_study import run_ablation_study

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LaGRAR - Task-aware Latent Graph Rewiring')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    
    # Experiment settings
    parser.add_argument('--task', type=str, default='node', choices=['node', 'link', 'graph'], 
                       help='Task type: node classification, link prediction, or graph classification')
    parser.add_argument('--dataset', type=str, default='cora', 
                       help='Dataset name (e.g., cora, citeseer, pubmed, etc.)')
    parser.add_argument('--dataset_type', type=str, default='real', choices=['real', 'synthetic'],
                       help='Type of dataset: real or synthetic')
    parser.add_argument('--synthetic_type', type=str, default='hom-sbm', 
                       choices=['hom-sbm', 'het-sbm', 'rcg', 'rhg'],
                       help='Type of synthetic dataset')
    
    # Model settings
    parser.add_argument('--model', type=str, default='lagrar', 
                       choices=['lagrar', 'gcn', 'gat', 'sdrf', 'fosr', 'borf', 'digl'],
                       help='Model to use')
    parser.add_argument('--encoder', type=str, default='gcn', choices=['gcn', 'gat'],
                       help='Encoder type for LaGRAR')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--latent_channels', type=int, default=32, help='Latent dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.5, help='CSRF parameter for positive curvature')
    parser.add_argument('--beta', type=float, default=0.5, help='CSRF parameter for negative curvature')
    parser.add_argument('--rho', type=float, default=0.1, help='Weight parameter for loss components')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    
    # Experiment specific
    parser.add_argument('--exp_id', type=str, default=None, help='Experiment ID')
    parser.add_argument('--run_ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--run_all_datasets', action='store_true', help='Run on all datasets for the task')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    return parser.parse_args()

def main():
    """Main function to run experiments."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    if args.verbose:
        print(f"Using device: {device}")
    
    # Generate experiment ID if not provided
    if args.exp_id is None:
        args.exp_id = f"{args.model}_{args.task}_{args.dataset}_{args.seed}"
    
    # Set up save paths
    model_save_path = os.path.join(args.save_dir, f"{args.exp_id}.pt")
    results_save_path = os.path.join(args.log_dir, f"{args.exp_id}_results.json")
    
    # Run the appropriate experiment
    if args.run_all_datasets:
        # Run experiment on all datasets for the task
        if args.task == 'node':
            datasets = ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'cornell', 'texas', 'wisconsin']
        elif args.task == 'link':
            datasets = ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'cornell', 'texas', 'wisconsin']
        elif args.task == 'graph':
            datasets = ['IMDB-BINARY', 'REDDIT-BINARY', 'MUTAG', 'PROTEINS', 'COLLAB', 'ENZYMES']
        
        results = {}
        for dataset in tqdm(datasets, desc="Datasets"):
            args.dataset = dataset
            # Update experiment ID
            args.exp_id = f"{args.model}_{args.task}_{args.dataset}_{args.seed}"
            model_save_path = os.path.join(args.save_dir, f"{args.exp_id}.pt")
            
            # Run experiment
            if args.task == 'node':
                result = run_node_classification_experiment(args, device, model_save_path)
            elif args.task == 'link':
                result = run_link_prediction_experiment(args, device, model_save_path)
            elif args.task == 'graph':
                result = run_graph_classification_experiment(args, device, model_save_path)
            
            results[dataset] = result
        
        # Save all results
        import json
        with open(results_save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\nResults Summary:")
        for dataset, result in results.items():
            print(f"{dataset}: {result['test_metrics']}")
    
    elif args.run_ablation:
        # Run ablation study
        run_ablation_study(args, device, model_save_path)
    
    else:
        # Run single experiment
        if args.task == 'node':
            result = run_node_classification_experiment(args, device, model_save_path)
        elif args.task == 'link':
            result = run_link_prediction_experiment(args, device, model_save_path)
        elif args.task == 'graph':
            result = run_graph_classification_experiment(args, device, model_save_path)
        
        # Save results
        import json
        with open(results_save_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Print summary
        print("\nResults:")
        print(f"Test metrics: {result['test_metrics']}")
        
        # Visualize if requested
        if args.visualize:
            # Create visualization directory
            vis_dir = os.path.join(args.log_dir, f"{args.exp_id}_visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Load the best model
            model = result['model']
            model.load_state_dict(torch.load(model_save_path))
            model.to(device)
            
            # Visualize loss components
            loss_history = result['loss_history']
            vis_path = os.path.join(vis_dir, "loss_components.png")
            visualize_loss_components(loss_history, save_path=vis_path)
            
            # Get the data
            data = result['data']
            
            # Visualize embeddings
            if args.task == 'node':
                labels = data.y.cpu().numpy()
                vis_path = os.path.join(vis_dir, "embeddings.png")
                with torch.no_grad():
                    outputs = model(data.x.to(device), data.edge_index.to(device))
                    z = outputs['z'] if isinstance(outputs, dict) else outputs
                visualize_embeddings(z.cpu().numpy(), labels, save_path=vis_path)
            
            # Visualize curvature distribution
            if hasattr(model, 'csrf'):
                vis_path = os.path.join(vis_dir, "curvature.png")
                curvature_results = evaluate_curvature_distribution(model, data, device)
                visualize_curvature_distribution(
                    curvature_results['original_curvature'], 
                    title="Original Graph Curvature",
                    save_path=vis_path
                )
                
                vis_path = os.path.join(vis_dir, "curvature_rewired.png")
                visualize_curvature_distribution(
                    curvature_results['rewired_curvature'], 
                    title="Rewired Graph Curvature",
                    save_path=vis_path
                )

if __name__ == '__main__':
    main()