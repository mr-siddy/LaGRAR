# models/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDiscriminator(nn.Module):
    """Hierarchical discriminator for adversarial CSRF training.
    
    This module implements the hierarchical discriminator that distinguishes
    between original graph embeddings and adversarial CSRF graph embeddings.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        """Initialize the hierarchical discriminator.
        
        Args:
            input_dim: Dimension of input node embeddings
            hidden_dims: Dimensions of hidden layers
        """
        super(HierarchicalDiscriminator, self).__init__()
        
        # First discriminator (for CSRF-1)
        self.disc1_layers = nn.ModuleList()
        self.disc1_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.disc1_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.disc1_layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Second discriminator (for CSRF-5)
        self.disc5_layers = nn.ModuleList()
        self.disc5_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.disc5_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.disc5_layers.append(nn.Linear(hidden_dims[-1], 1))
    
    def forward_disc1(self, z):
        """Forward pass through the first discriminator.
        
        Args:
            z: Node embeddings of shape [num_nodes, input_dim]
            
        Returns:
            Discriminator predictions for CSRF-1
        """
        h = z
        for i, layer in enumerate(self.disc1_layers):
            h = layer(h)
            if i < len(self.disc1_layers) - 1:
                h = F.leaky_relu(h, negative_slope=0.2)
        
        return torch.sigmoid(h)
    
    def forward_disc5(self, z):
        """Forward pass through the second discriminator.
        
        Args:
            z: Node embeddings of shape [num_nodes, input_dim]
            
        Returns:
            Discriminator predictions for CSRF-5
        """
        h = z
        for i, layer in enumerate(self.disc5_layers):
            h = layer(h)
            if i < len(self.disc5_layers) - 1:
                h = F.leaky_relu(h, negative_slope=0.2)
        
        return torch.sigmoid(h)
    
    def forward(self, z, discriminator_id=None):
        """Forward pass through the hierarchical discriminator.
        
        Args:
            z: Node embeddings of shape [num_nodes, input_dim]
            discriminator_id: Which discriminator to use (1, 5, or None for both)
            
        Returns:
            Discriminator predictions
        """
        if discriminator_id == 1:
            return self.forward_disc1(z)
        elif discriminator_id == 5:
            return self.forward_disc5(z)
        else:
            # Return outputs from both discriminators
            return self.forward_disc1(z), self.forward_disc5(z)
    
    def get_disc_loss(self, z_original, z_csrf1, z_csrf5):
        """Compute the hierarchical adversarial loss.
        
        Args:
            z_original: Original graph embeddings
            z_csrf1: CSRF-1 graph embeddings
            z_csrf5: CSRF-5 graph embeddings
            
        Returns:
            Total discriminator loss
        """
        # Discriminator 1 loss
        real_pred1 = self.forward_disc1(z_original)
        fake_pred1 = self.forward_disc1(z_csrf1)
        
        disc1_loss = -torch.mean(torch.log(real_pred1 + 1e-10) + torch.log(1 - fake_pred1 + 1e-10))
        
        # Discriminator 5 loss
        real_pred5 = self.forward_disc5(z_original)
        fake_pred5 = self.forward_disc5(z_csrf5)
        
        disc5_loss = -torch.mean(torch.log(real_pred5 + 1e-10) + torch.log(1 - fake_pred5 + 1e-10))
        
        # Total loss (weighted as in the paper: ρ² · LDi + ρ · LDj)
        rho = 0.1  # As specified in the paper
        total_loss = (rho**2) * disc1_loss + rho * disc5_loss
        
        return total_loss