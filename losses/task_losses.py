"""
Task-specific loss functions for LaGRAR.

This module implements the task-specific losses described in Eq. 5 of the paper:
1. Node Classification Loss
2. Link Prediction Loss
3. Graph Classification Loss

It also implements a regularization loss used for graph classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict

class NodeClassificationLoss(nn.Module):
    """
    Node classification loss as described in Eq. 5 of the paper.
    
    Formula: Lnode = -(1/N) * ∑(i=1 to N) ∑(c=1 to C) yi,c * log(ŷi,c)
    
    Where:
    - N: Number of nodes
    - C: Number of classes
    - yi,c: Ground truth label
    - ŷi,c: Predicted probability
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize the loss function.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(NodeClassificationLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the node classification loss.
        
        Args:
            pred: Predicted logits of shape [num_nodes, num_classes]
            target: Ground truth labels of shape [num_nodes]
            mask: Optional mask for nodes to compute loss on
            
        Returns:
            Classification loss
        """
        if mask is not None:
            # Apply mask to predictions and targets
            pred = pred[mask]
            target = target[mask]
        
        # Compute cross entropy loss
        loss = F.cross_entropy(pred, target, reduction=self.reduction)
        
        return loss

class LinkPredictionLoss(nn.Module):
    """
    Link prediction loss as described in Eq. 5 of the paper.
    
    Formula: Llink = -(1/(N(N-1))) * ∑(i=1 to N) ∑(j=1 to N, j≠i) [yij * log(ŷij) + (1-yij) * log(1-ŷij)]
    
    Where:
    - N: Number of nodes
    - yij: Ground truth for edge (i,j)
    - ŷij: Predicted probability for edge (i,j)
    """
    
    def __init__(self, pos_weight: Optional[float] = None, reduction: str = 'mean'):
        """
        Initialize the loss function.
        
        Args:
            pos_weight: Weight for positive examples
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(LinkPredictionLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, 
                pred: torch.Tensor, 
                pos_edge_index: torch.Tensor, 
                neg_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute the link prediction loss.
        
        Args:
            pred: Edge predictions of shape [num_edges]
            pos_edge_index: Positive edge indices of shape [2, num_pos_edges]
            neg_edge_index: Negative edge indices of shape [2, num_neg_edges]
            
        Returns:
            Link prediction loss
        """
        # Create target vectors (1 for positive edges, 0 for negative edges)
        pos_target = torch.ones(pos_edge_index.size(1), device=pred.device)
        neg_target = torch.zeros(neg_edge_index.size(1), device=pred.device)
        
        # Get predictions for positive and negative edges
        pos_pred = pred[:pos_edge_index.size(1)]
        neg_pred = pred[pos_edge_index.size(1):]
        
        # Combine predictions and targets
        all_pred = torch.cat([pos_pred, neg_pred], dim=0)
        all_target = torch.cat([pos_target, neg_target], dim=0)
        
        # Compute binary cross entropy loss
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=pred.device)
            loss = F.binary_cross_entropy_with_logits(
                all_pred, all_target, 
                pos_weight=pos_weight,
                reduction=self.reduction
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                all_pred, all_target, 
                reduction=self.reduction
            )
        
        return loss

class GraphClassificationLoss(nn.Module):
    """
    Graph classification loss as described in Eq. 5 of the paper.
    
    Formula: Lgraph = -∑(c=1 to C) yc * log(ŷc)
    
    Where:
    - C: Number of classes
    - yc: Ground truth label
    - ŷc: Predicted probability
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize the loss function.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(GraphClassificationLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the graph classification loss.
        
        Args:
            pred: Predicted logits of shape [batch_size, num_classes]
            target: Ground truth labels of shape [batch_size]
            
        Returns:
            Classification loss
        """
        return F.cross_entropy(pred, target, reduction=self.reduction)

class RegularizationLoss(nn.Module):
    """
    Regularization loss for graph embeddings to address over-smoothing.
    
    This loss is used in graph classification to ensure that graph embeddings
    remain distinguishable.
    
    Formula: Lgraph_reg = -(1/N) * ∑(i=1 to N) log(exp(gi·gj)/(∑(j=1 to N) exp(gi·gj)))
    
    Where:
    - N: Number of graphs
    - gi, gj: Graph embeddings
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize the loss function.
        
        Args:
            temperature: Temperature parameter for scaling similarities
        """
        super(RegularizationLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the regularization loss.
        
        Args:
            embeddings: Graph embeddings of shape [batch_size, embedding_dim]
            
        Returns:
            Regularization loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Remove self-similarities
        batch_size = embeddings.size(0)
        mask = torch.ones_like(similarity_matrix) - torch.eye(batch_size, device=embeddings.device)
        masked_similarities = similarity_matrix * mask
        
        # Compute regularization loss
        exp_sim = torch.exp(masked_similarities)
        loss = -torch.mean(torch.log(exp_sim.sum(dim=1) / exp_sim.sum()))
        
        return loss