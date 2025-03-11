# models/lagrar.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, negative_sampling

from .encoders import GCNEncoder, GATEncoder
from .decoders import InnerProductDecoder, LocalityPreservingDecoder
from .discriminator import HierarchicalDiscriminator
from ..modules.csrf import CurvatureSensitiveRicciFlow
from ..modules.latent_substructure import LatentSubstructureComputation
from ..losses.positive_loss import PositiveLoss
from ..losses.negative_loss import NegativeLoss
from ..losses.reconstruction_loss import ReconstructionLoss
from ..losses.task_losses import NodeClassificationLoss, LinkPredictionLoss, GraphClassificationLoss

class LaGRAR(nn.Module):
    """LaGRAR: LGR with Hierarchical Adversarial Ricci Flows and Walks in Latent Space.
    
    This is the main implementation of the LaGRAR model as described in the paper.
    """
    
    def __init__(self, in_channels, hidden_channels, latent_channels, 
                 encoder_type='gcn', task_type='node', num_classes=None, 
                 dropout=0.5, alpha=0.5, beta=0.5, rho=0.1):
        """Initialize the LaGRAR model.
        
        Args:
            in_channels: Dimension of input node features
            hidden_channels: Dimension of hidden layers
            latent_channels: Dimension of latent space
            encoder_type: Type of encoder ('gcn' or 'gat')
            task_type: Type of task ('node', 'link', or 'graph')
            num_classes: Number of classes for classification tasks
            dropout: Dropout probability
            alpha: CSRF parameter for positive curvature sensitivity
            beta: CSRF parameter for negative curvature sensitivity
            rho: Weight parameter for loss components
        """
        super(LaGRAR, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.encoder_type = encoder_type
        self.task_type = task_type
        self.num_classes = num_classes
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        
        # Initialize encoder based on type
        if encoder_type == 'gcn':
            self.encoder = GCNEncoder(in_channels, hidden_channels, latent_channels, dropout)
        elif encoder_type == 'gat':
            self.encoder = GATEncoder(in_channels, hidden_channels, latent_channels, heads=8, dropout=dropout)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # Initialize locality preserving decoder
        self.decoder = LocalityPreservingDecoder(dropout)
        
        # Initialize CSRF module
        self.csrf = CurvatureSensitiveRicciFlow(alpha, beta)
        
        # Initialize latent substructure computation
        self.substructure = LatentSubstructureComputation()
        
        # Initialize hierarchical discriminator
        self.discriminator = HierarchicalDiscriminator(latent_channels)
        
        # Initialize task-specific head
        if task_type == 'node':
            self.task_head = nn.Linear(latent_channels, num_classes)
        elif task_type == 'link':
            self.task_head = nn.Sequential(
                nn.Linear(latent_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1)
            )
        elif task_type == 'graph':
            self.task_head = nn.Sequential(
                nn.Linear(latent_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes)
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Initialize loss functions
        self.positive_loss = PositiveLoss(rho)
        self.negative_loss = NegativeLoss(rho)
        self.reconstruction_loss = ReconstructionLoss()
        
        if task_type == 'node':
            self.task_loss = NodeClassificationLoss()
        elif task_type == 'link':
            self.task_loss = LinkPredictionLoss()
        elif task_type == 'graph':
            self.task_loss = GraphClassificationLoss()
    
    def encode(self, x, edge_index, edge_weight=None):
        """Encode node features to latent representations.
        
        Args:
            x: Node features of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            z: Sampled node embeddings of shape [num_nodes, latent_channels]
            mu: Mean vectors of shape [num_nodes, latent_channels]
            logvar: Log variance vectors of shape [num_nodes, latent_channels]
        """
        mu, logvar = self.encoder(x, edge_index, edge_weight)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        return z, mu, logvar
    
    def decode(self, z, edge_index=None):
        """Decode latent representations to reconstruct the graph.
        
        Args:
            z: Node embeddings of shape [num_nodes, latent_channels]
            edge_index: Optional edge indices for which to compute probabilities
            
        Returns:
            Reconstructed adjacency matrix or edge probabilities
        """
        return self.decoder(z, edge_index)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """Forward pass through the LaGRAR model.
        
        Args:
            x: Node features of shape [num_nodes, in_channels]
            edge_index: Edge indices of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            batch: Batch indices for graph-level tasks
            
        Returns:
            Dictionary with model outputs based on task type
        """
        num_nodes = x.size(0)
        
        # Encode original graph
        z, mu, logvar = self.encode(x, edge_index, edge_weight)
        
        # Generate CSRF adversarial graphs
        edge_index_csrf1, edge_weight_csrf1 = self.csrf.apply_flow(edge_index, edge_weight, num_nodes, iterations=1)
        edge_index_csrf5, edge_weight_csrf5 = self.csrf.apply_flow(edge_index, edge_weight, num_nodes, iterations=5)
        
        # Encode CSRF adversarial graphs
        z_csrf1, _, _ = self.encode(x, edge_index_csrf1, edge_weight_csrf1)
        z_csrf5, _, _ = self.encode(x, edge_index_csrf5, edge_weight_csrf5)
        
        # Decode to reconstruct adjacency
        adj_pred = self.decode(z)
        
        # Task-specific predictions
        if self.task_type == 'node':
            task_pred = self.task_head(z)
        elif self.task_type == 'link':
            # For link prediction, this will be computed in the loss function
            task_pred = z  # Just return embeddings
        elif self.task_type == 'graph':
            # Graph-level pooling and prediction
            if batch is None:
                # If batch indices not provided, assume single graph
                graph_embedding = z.mean(dim=0, keepdim=True)
            else:
                # Mean pooling across nodes in each graph
                graph_embedding = torch.zeros(batch.max().item() + 1, z.size(1), device=z.device)
                graph_counts = torch.zeros(batch.max().item() + 1, device=z.device)
                
                for i, b in enumerate(batch):
                    graph_embedding[b] += z[i]
                    graph_counts[b] += 1
                
                graph_embedding = graph_embedding / graph_counts.unsqueeze(1).clamp(min=1)
            
            task_pred = self.task_head(graph_embedding)
        
        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'z_csrf1': z_csrf1,
            'z_csrf5': z_csrf5,
            'adj_pred': adj_pred,
            'task_pred': task_pred
        }
    
    def get_loss(self, outputs, x, edge_index, edge_weight=None, y=None, batch=None):
        """Compute the total loss for LaGRAR.
        
        Args:
            outputs: Dictionary of model outputs from forward pass
            x: Node features
            edge_index: Edge indices
            edge_weight: Optional edge weights
            y: Target labels
            batch: Batch indices for graph-level tasks
            
        Returns:
            Dictionary of loss components and total loss
        """
        z = outputs['z']
        mu = outputs['mu']
        logvar = outputs['logvar']
        z_csrf1 = outputs['z_csrf1']
        z_csrf5 = outputs['z_csrf5']
        adj_pred = outputs['adj_pred']
        task_pred = outputs['task_pred']
        
        # Compute substructure counts for positive loss
        alpha, beta, gamma = self.substructure.compute_all(z, edge_index)
        
        # Compute positive loss (substructure loss)
        pos_loss = self.positive_loss(alpha, beta, gamma, edge_index)
        
        # Compute negative loss (adversarial CSRF loss)
        neg_loss = self.negative_loss(z, z_csrf1, z_csrf5, self.discriminator)
        
        # Compute reconstruction loss
        adj = to_dense_adj(edge_index)[0]  # Convert to dense adjacency matrix
        rec_loss = self.reconstruction_loss(adj_pred, adj)
        
        # Compute KL divergence loss
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # Compute task-specific loss
        if self.task_type == 'node':
            if y is None:
                raise ValueError("Target labels required for node classification")
            task_loss = self.task_loss(task_pred, y)
        elif self.task_type == 'link':
            if y is None:
                # If no specific links provided, use all edges
                pos_edge_index = edge_index
                # Sample negative edges
                neg_edge_index = negative_sampling(edge_index, num_nodes=z.size(0))
                y_true = torch.cat([torch.ones(pos_edge_index.size(1)), 
                                   torch.zeros(neg_edge_index.size(1))], dim=0)
                edge_index_all = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            else:
                # Use provided links and labels
                edge_index_all = y['edge_index']
                y_true = y['labels']
            
            # Compute link probabilities
            src, dst = edge_index_all
            link_logits = torch.sum(z[src] * z[dst], dim=1)
            task_loss = self.task_loss(link_logits, y_true)
        elif self.task_type == 'graph':
            if y is None:
                raise ValueError("Target labels required for graph classification")
            task_loss = self.task_loss(task_pred, y)
        
        # Compute total loss as per Eq. 4 in the paper
        total_loss = rec_loss + kl_loss + pos_loss + neg_loss + task_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': rec_loss,
            'kl_loss': kl_loss,
            'positive_loss': pos_loss,
            'negative_loss': neg_loss,
            'task_loss': task_loss
        }
    
    def predict(self, x, edge_index, edge_weight=None, batch=None):
        """Make predictions with the LaGRAR model.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Optional edge weights
            batch: Batch indices for graph-level tasks
            
        Returns:
            Predictions based on task type
        """
        outputs = self(x, edge_index, edge_weight, batch)
        
        if self.task_type == 'node':
            return F.softmax(outputs['task_pred'], dim=1)
        elif self.task_type == 'link':
            z = outputs['z']
            # For link prediction, compute pairwise similarities
            adj_pred = torch.matmul(z, z.t())
            return torch.sigmoid(adj_pred)
        elif self.task_type == 'graph':
            return F.softmax(outputs['task_pred'], dim=1)