# utils/evaluation.py (continued)
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch_geometric.utils import negative_sampling, to_undirected

def evaluate_node_classification(model, data, device, mask=None):
    """Evaluate the model on node classification.
    
    Args:
        model: Trained model
        data: Input data
        device: Device to use
        mask: Optional mask for node subset evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, 'edge_weight') else None
    y = data.y.to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(x, edge_index, edge_weight)
        
        # Get predictions
        if isinstance(outputs, dict):
            # LaGRAR model
            logits = outputs['task_pred']
        else:
            # Base model
            logits = outputs
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.to(device)
            logits = logits[mask]
            y = y[mask]
        
        # Convert to probabilities and predicted classes
        if logits.size(1) > 1:  # Multi-class
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        else:  # Binary
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
    
    # Move to CPU for evaluation
    probs = probs.cpu().numpy()
    preds = preds.cpu().numpy()
    y = y.cpu().numpy()
    
    # Compute metrics
    acc = accuracy_score(y, preds)
    
    # Check for multi-class classification
    if len(np.unique(y)) > 2:
        prec = precision_score(y, preds, average='macro')
        rec = recall_score(y, preds, average='macro')
        f1_val = f1_score(y, preds, average='macro')
        
        # One-hot encode for AUC
        n_classes = len(np.unique(y))
        y_onehot = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            y_onehot[i, label] = 1
        
        try:
            auc = roc_auc_score(y_onehot, probs, multi_class='ovr')
        except ValueError:
            auc = np.nan
    else:
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        f1_val = f1_score(y, preds)
        
        try:
            auc = roc_auc_score(y, probs)
        except ValueError:
            auc = np.nan
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1_val,
        'auc': auc
    }

def evaluate_link_prediction(model, data, device, test_pos_edge_index=None, num_neg_samples=None):
    """Evaluate the model on link prediction.
    
    Args:
        model: Trained model
        data: Input data
        device: Device to use
        test_pos_edge_index: Optional test positive edge indices
        num_neg_samples: Number of negative samples
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, 'edge_weight') else None
    
    # Use test edges from data if not provided
    if test_pos_edge_index is None:
        if hasattr(data, 'test_pos_edge_index'):
            test_pos_edge_index = data.test_pos_edge_index.to(device)
        else:
            raise ValueError("Test positive edges not provided")
    else:
        test_pos_edge_index = test_pos_edge_index.to(device)
    
    # Generate negative samples if needed
    if num_neg_samples is None:
        num_neg_samples = test_pos_edge_index.size(1)
    
    test_neg_edge_index = negative_sampling(
        edge_index=to_undirected(edge_index),
        num_nodes=data.num_nodes,
        num_neg_samples=num_neg_samples
    ).to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(x, edge_index, edge_weight)
        
        # Get embeddings
        if isinstance(outputs, dict):
            # LaGRAR model
            z = outputs['z']
        else:
            # Base model
            z = outputs
        
        # Compute predictions for positive edges
        pos_scores = []
        for i in range(0, test_pos_edge_index.size(1), 1000):  # Process in batches
            batch_edge_index = test_pos_edge_index[:, i:i+1000]
            src, dst = batch_edge_index
            pos_scores.append(torch.sum(z[src] * z[dst], dim=1))
        pos_scores = torch.cat(pos_scores)
        
        # Compute predictions for negative edges
        neg_scores = []
        for i in range(0, test_neg_edge_index.size(1), 1000):  # Process in batches
            batch_edge_index = test_neg_edge_index[:, i:i+1000]
            src, dst = batch_edge_index
            neg_scores.append(torch.sum(z[src] * z[dst], dim=1))
        neg_scores = torch.cat(neg_scores)
    
    # Move to CPU for evaluation
    pos_scores = pos_scores.cpu().numpy()
    neg_scores = neg_scores.cpu().numpy()
    
    # Create labels (1 for positive edges, 0 for negative)
    y_true = np.concatenate([np.ones(pos_scores.shape[0]), np.zeros(neg_scores.shape[0])])
    y_pred = np.concatenate([pos_scores, neg_scores])
    
    # Compute metrics
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = np.nan
    
    # Convert scores to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary)
    rec = recall_score(y_true, y_pred_binary)
    f1_val = f1_score(y_true, y_pred_binary)
    
    return {
        'auc': auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1_val
    }

def evaluate_graph_classification(model, data_loader, device):
    """Evaluate the model on graph classification.
    
    Args:
        model: Trained model
        data_loader: DataLoader with test graphs
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # Forward pass
            outputs = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            # Get predictions
            if isinstance(outputs, dict):
                # LaGRAR model
                logits = outputs['task_pred']
            else:
                # Base model
                logits = outputs
            
            # Convert to probabilities and predicted classes
            if logits.size(1) > 1:  # Multi-class
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
            else:  # Binary
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            
            # Collect predictions and labels
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    
    # Concatenate results
    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    
    # Check for multi-class classification
    if len(np.unique(all_labels)) > 2:
        prec = precision_score(all_labels, all_preds, average='macro')
        rec = recall_score(all_labels, all_preds, average='macro')
        f1_val = f1_score(all_labels, all_preds, average='macro')
        
        # One-hot encode for AUC
        n_classes = all_probs.shape[1]
        y_onehot = np.zeros((len(all_labels), n_classes))
        for i, label in enumerate(all_labels):
            y_onehot[i, label] = 1
        
        try:
            auc = roc_auc_score(y_onehot, all_probs, multi_class='ovr')
        except ValueError:
            auc = np.nan
    else:
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1_val = f1_score(all_labels, all_preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs)
        except ValueError:
            auc = np.nan
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1_val,
        'auc': auc
    }

def evaluate_curvature_distribution(model, data, device):
    """Evaluate the curvature distribution before and after rewiring.
    
    Args:
        model: Trained LaGRAR model
        data: Input data
        device: Device to use
        
    Returns:
        Dictionary with curvature statistics
    """
    from ..modules.curvature import compute_ollivier_ricci_curvature
    from torch_geometric.utils import to_networkx
    
    model.eval()
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, 'edge_weight') else None
    
    # Get original graph as NetworkX
    G_original = to_networkx(data, to_undirected=True)
    
    # Compute original curvature
    original_curvature = compute_ollivier_ricci_curvature(G_original)
    
    with torch.no_grad():
        # Forward pass to get embeddings
        outputs = model(x, edge_index, edge_weight)
        
        # Get embeddings
        z = outputs['z']
        
        # Reconstruct graph from embeddings
        adj_pred = model.decode(z)
        
        # Convert to binary adjacency matrix with threshold
        adj_pred_binary = (adj_pred > 0.5).cpu().numpy()
        
        # Create rewired graph
        num_nodes = adj_pred_binary.shape[0]
        G_rewired = nx.Graph()
        G_rewired.add_nodes_from(range(num_nodes))
        
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_pred_binary[i, j]:
                    edges.append((i, j))
        
        G_rewired.add_edges_from(edges)
        
        # Compute rewired curvature
        rewired_curvature = compute_ollivier_ricci_curvature(G_rewired)
    
    # Compute statistics
    original_values = list(original_curvature.values())
    rewired_values = list(rewired_curvature.values())
    
    original_stats = {
        'mean': np.mean(original_values),
        'std': np.std(original_values),
        'min': np.min(original_values),
        'max': np.max(original_values),
        'positive_ratio': np.mean(np.array(original_values) > 0),
        'negative_ratio': np.mean(np.array(original_values) < 0),
        'zero_ratio': np.mean(np.array(original_values) == 0)
    }
    
    rewired_stats = {
        'mean': np.mean(rewired_values),
        'std': np.std(rewired_values),
        'min': np.min(rewired_values),
        'max': np.max(rewired_values),
        'positive_ratio': np.mean(np.array(rewired_values) > 0),
        'negative_ratio': np.mean(np.array(rewired_values) < 0),
        'zero_ratio': np.mean(np.array(rewired_values) == 0)
    }
    
    return {
        'original': original_stats,
        'rewired': rewired_stats,
        'original_curvature': original_curvature,
        'rewired_curvature': rewired_curvature
    }

def evaluate_lagrar_components(model, data, device, task_type='node'):
    """Evaluate the contribution of different LaGRAR components.
    
    Args:
        model: Trained LaGRAR model
        data: Input data
        device: Device to use
        task_type: Type of task ('node', 'link', 'graph')
        
    Returns:
        Dictionary with component contributions
    """
    model.eval()
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, 'edge_weight') else None
    
    # Get target labels based on task type
    if task_type == 'node':
        y = data.y.to(device)
        batch = None
        mask = data.test_mask.to(device) if hasattr(data, 'test_mask') else None
    elif task_type == 'link':
        if hasattr(data, 'test_pos_edge_index'):
            test_pos_edge_index = data.test_pos_edge_index.to(device)
        else:
            raise ValueError("Test positive edges not provided")
        y = None
        batch = None
    elif task_type == 'graph':
        y = data.y.to(device)
        batch = data.batch.to(device) if hasattr(data, 'batch') else None
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Full model evaluation
    with torch.no_grad():
        outputs = model(x, edge_index, edge_weight, batch)
        
    if task_type == 'node':
        full_model_metrics = evaluate_node_classification(model, data, device, mask)
    elif task_type == 'link':
        full_model_metrics = evaluate_link_prediction(model, data, device, test_pos_edge_index)
    elif task_type == 'graph':
        # For graph tasks, we need a DataLoader
        raise ValueError("Graph task evaluation requires a DataLoader, use evaluate_graph_classification instead")
    
    # Ablation studies
    ablation_results = {'full_model': full_model_metrics}
    
    # Disable positive loss (substructure computation)
    original_positive_loss = model.positive_loss
    model.positive_loss = lambda *args, **kwargs: torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        outputs_no_pos = model(x, edge_index, edge_weight, batch)
    
    if task_type == 'node':
        ablation_results['no_positive_loss'] = evaluate_node_classification(model, data, device, mask)
    elif task_type == 'link':
        ablation_results['no_positive_loss'] = evaluate_link_prediction(model, data, device, test_pos_edge_index)
    
    # Restore positive loss
    model.positive_loss = original_positive_loss
    
    # Disable negative loss (adversarial CSRF)
    original_negative_loss = model.negative_loss
    model.negative_loss = lambda *args, **kwargs: torch.tensor(0.0, device=device)
    
    with torch.no_grad():
        outputs_no_neg = model(x, edge_index, edge_weight, batch)
    
    if task_type == 'node':
        ablation_results['no_negative_loss'] = evaluate_node_classification(model, data, device, mask)
    elif task_type == 'link':
        ablation_results['no_negative_loss'] = evaluate_link_prediction(model, data, device, test_pos_edge_index)
    
    # Restore negative loss
    model.negative_loss = original_negative_loss
    
    # Disable locality preserving sparsification
    original_decoder = model.decoder
    from ..models.decoders import InnerProductDecoder
    model.decoder = InnerProductDecoder(model.dropout)
    
    with torch.no_grad():
        outputs_no_sparsification = model(x, edge_index, edge_weight, batch)
    
    if task_type == 'node':
        ablation_results['no_sparsification'] = evaluate_node_classification(model, data, device, mask)
    elif task_type == 'link':
        ablation_results['no_sparsification'] = evaluate_link_prediction(model, data, device, test_pos_edge_index)
    
    # Restore decoder
    model.decoder = original_decoder
    
    return ablation_results