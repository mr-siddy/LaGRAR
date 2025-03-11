# modules/latent_substructure.py
import torch
import torch.nn.functional as F

class LatentSubstructureComputation:
    """Compute substructures in the latent space.
    
    This module implements the computation of triangles, quadrangles,
    and pentagons in the latent space as described in the paper.
    """
    
    def __init__(self):
        """Initialize the latent substructure computation module."""
        pass
    
    def compute_triangles(self, z, edge_index):
        """Compute the number of triangles based at each edge.
        
        Args:
            z: Node embeddings of shape [num_nodes, hidden_dim]
            edge_index: Edge indices of shape [2, num_edges]
            
        Returns:
            Triangle counts for each edge
        """
        src, dst = edge_index
        triangles = []
        
        for i in range(edge_index.size(1)):
            u, v = src[i].item(), dst[i].item()
            
            # Get neighbors of u
            u_neighbors = src[dst == u]
            
            # Compute triangle count using normalized dot products
            alpha = 0.0
            for u_prime in u_neighbors:
                if u_prime != v:  # Exclude the edge (u, v)
                    z_u_prime = z[u_prime]
                    z_v = z[v]
                    
                    # Normalized dot product
                    sim = F.cosine_similarity(z_u_prime.unsqueeze(0), z_v.unsqueeze(0), dim=1)
                    alpha += sim.item()
            
            triangles.append(alpha)
        
        return torch.tensor(triangles, device=z.device)
    
    def compute_quadrangles(self, z, edge_index):
        """Compute the number of quadrangles based at each edge.
        
        Args:
            z: Node embeddings of shape [num_nodes, hidden_dim]
            edge_index: Edge indices of shape [2, num_edges]
            
        Returns:
            Quadrangle counts for each edge
        """
        src, dst = edge_index
        quadrangles = []
        
        for i in range(edge_index.size(1)):
            u, v = src[i].item(), dst[i].item()
            
            # Get neighbors of u and v
            u_neighbors = src[dst == u]
            v_neighbors = src[dst == v]
            
            # Compute quadrangle count
            beta = 0.0
            for u_prime in u_neighbors:
                if u_prime != v:  # Exclude the edge (u, v)
                    for v_prime in v_neighbors:
                        if v_prime != u and v_prime != u_prime:  # Exclude trivial cycles
                            z_u_prime = z[u_prime]
                            z_v_prime = z[v_prime]
                            
                            # Normalized dot product
                            sim = F.cosine_similarity(z_u_prime.unsqueeze(0), z_v_prime.unsqueeze(0), dim=1)
                            beta += sim.item()
            
            quadrangles.append(beta)
        
        return torch.tensor(quadrangles, device=z.device)
    
    def compute_pentagons(self, z, edge_index):
        """Compute the number of pentagons based at each edge.
        
        Args:
            z: Node embeddings of shape [num_nodes, hidden_dim]
            edge_index: Edge indices of shape [2, num_edges]
            
        Returns:
            Pentagon counts for each edge
        """
        src, dst = edge_index
        pentagons = []
        
        for i in range(edge_index.size(1)):
            u, v = src[i].item(), dst[i].item()
            
            # Get neighbors of u and v
            u_neighbors = src[dst == u]
            v_neighbors = src[dst == v]
            
            # Compute pentagon count
            gamma = 0.0
            for u_prime in u_neighbors:
                if u_prime != v:  # Exclude the edge (u, v)
                    for v_prime in v_neighbors:
                        if v_prime != u and v_prime != u_prime:  # Exclude trivial cycles
                            # Get neighbors of v_prime
                            v_prime_neighbors = src[dst == v_prime]
                            
                            for w in v_prime_neighbors:
                                if w != v and w != u and w != u_prime:  # Exclude trivial cycles
                                    z_u_prime = z[u_prime]
                                    z_w = z[w]
                                    
                                    # Normalized dot product
                                    sim = F.cosine_similarity(z_u_prime.unsqueeze(0), z_w.unsqueeze(0), dim=1)
                                    gamma += sim.item()
            
            pentagons.append(gamma)
        
        return torch.tensor(pentagons, device=z.device)
    
    def compute_all(self, z, edge_index):
        """Compute all substructures (triangles, quadrangles, pentagons).
        
        Args:
            z: Node embeddings of shape [num_nodes, hidden_dim]
            edge_index: Edge indices of shape [2, num_edges]
            
        Returns:
            Tuple of (triangles, quadrangles, pentagons)
        """
        alpha = self.compute_triangles(z, edge_index)
        beta = self.compute_quadrangles(z, edge_index)
        gamma = self.compute_pentagons(z, edge_index)
        
        return alpha, beta, gamma