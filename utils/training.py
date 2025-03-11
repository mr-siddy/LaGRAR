# utils/training.py
import time
import torch
import numpy as np
from tqdm import tqdm

class EarlyStopping:
    """Early stopping to terminate training when validation loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0, verbose=True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss, model, path=None):
        """Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if validation loss improves
            path: Path to save the model
            
        Returns:
            True if training should be stopped, False otherwise
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, val_loss, model, path=None):
        """Save model when validation loss decreases.
        
        Args:
            val_loss: Current validation loss
            model: Model to save
            path: Path to save the model
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        if path is not None:
            torch.save(model.state_dict(), path)
        
        self.val_loss_min = val_loss

def train_epoch(model, optimizer, data, device, task_type='node'):
    """Train the model for one epoch.
    
    Args:
        model: Model to train
        optimizer: Optimizer to use
        data: Input data
        device: Device to use
        task_type: Type of task ('node', 'link', 'graph')
        
    Returns:
        Dictionary with training losses
    """
    model.train()
    optimizer.zero_grad()
    
    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if hasattr(data, 'edge_weight') else None
    
    # Get target labels based on task type
    if task_type == 'node':
        y = data.y.to(device)
        batch = None
    elif task_type == 'link':
        y = {'edge_index': data.train_pos_edge_index.to(device), 
             'labels': torch.ones(data.train_pos_edge_index.size(1)).to(device)}
        batch = None
    elif task_type == 'graph':
        y = data.y.to(device)
        batch = data.batch.to(device) if hasattr(data, 'batch') else None
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Forward pass
    outputs = model(x, edge_index, edge_weight, batch)
    
    # Compute loss
    loss_dict = model.get_loss(outputs, x, edge_index, edge_weight, y, batch)
    loss = loss_dict['total_loss']
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Convert loss dictionary to numpy
    loss_dict_np = {k: v.item() for k, v in loss_dict.items()}
    
    return loss_dict_np

def validate(model, data, device, task_type='node'):
    """Validate the model.
    
    Args:
        model: Model to validate
        data: Validation data
        device: Device to use
        task_type: Type of task ('node', 'link', 'graph')
        
    Returns:
        Dictionary with validation losses
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
    elif task_type == 'link':
        y = {'edge_index': data.val_pos_edge_index.to(device), 
             'labels': torch.ones(data.val_pos_edge_index.size(1)).to(device)}
        batch = None
    elif task_type == 'graph':
        y = data.y.to(device)
        batch = data.batch.to(device) if hasattr(data, 'batch') else None
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    with torch.no_grad():
        # Forward pass
        outputs = model(x, edge_index, edge_weight, batch)
        
        # Compute loss
        loss_dict = model.get_loss(outputs, x, edge_index, edge_weight, y, batch)
    
    # Convert loss dictionary to numpy
    loss_dict_np = {k: v.item() for k, v in loss_dict.items()}
    
    return loss_dict_np

def train(model, optimizer, data, device, num_epochs=200, patience=20, 
          task_type='node', verbose=True, save_path=None):
    """Train the model.
    
    Args:
        model: Model to train
        optimizer: Optimizer to use
        data: Input data
        device: Device to use
        num_epochs: Number of epochs to train
        patience: Patience for early stopping
        task_type: Type of task ('node', 'link', 'graph')
        verbose: Whether to print progress
        save_path: Path to save the best model
        
    Returns:
        Trained model and history of losses
    """
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)
    
    # Initialize loss history
    train_loss_history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'kl_loss': [],
        'positive_loss': [],
        'negative_loss': [],
        'task_loss': []
    }
    
    val_loss_history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'kl_loss': [],
        'positive_loss': [],
        'negative_loss': [],
        'task_loss': []
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training step
        train_loss_dict = train_epoch(model, optimizer, data, device, task_type)
        
        # Update training loss history
        for k, v in train_loss_dict.items():
            if k in train_loss_history:
                train_loss_history[k].append(v)
        
        # Validation step
        val_loss_dict = validate(model, data, device, task_type)
        
        # Update validation loss history
        for k, v in val_loss_dict.items():
            if k in val_loss_history:
                val_loss_history[k].append(v)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {train_loss_dict["total_loss"]:.4f}, '
                  f'Val Loss: {val_loss_dict["total_loss"]:.4f}, '
                  f'Time: {elapsed_time:.2f}s')
        
        # Early stopping
        if early_stopping(val_loss_dict['total_loss'], model, save_path):
            if verbose:
                print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model if saved
    if save_path is not None:
        model.load_state_dict(torch.load(save_path))
    
    return model, {'train': train_loss_history, 'val': val_loss_history}