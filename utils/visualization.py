# utils/visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_geometric.utils import to_networkx

def visualize_graph(edge_index, node_features=None, node_labels=None, node_pos=None, edge_weights=None, title=None, figsize=(10, 10), save_path=None):
    """Visualize a graph with optional node features, labels, and edge weights.
    
    Args:
        edge_index: PyTorch Geometric edge index
        node_features: Optional node feature matrix
        node_labels: Optional node labels
        node_pos: Optional node positions (if None, layout is computed)
        edge_weights: Optional edge weights
        title: Optional plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure
    """
    # Convert to NetworkX graph
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    
    edge_list = list(zip(edge_index[0], edge_index[1]))
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Set node attributes
    if node_features is not None:
        if isinstance(node_features, torch.Tensor):
            node_features = node_features.cpu().numpy()
        for i, features in enumerate(node_features):
            G.nodes[i]['features'] = features
    
    if node_labels is not None:
        if isinstance(node_labels, torch.Tensor):
            node_labels = node_labels.cpu().numpy()
        for i, label in enumerate(node_labels):
            G.nodes[i]['label'] = label
    
    # Set edge weights
    if edge_weights is not None:
        if isinstance(edge_weights, torch.Tensor):
            edge_weights = edge_weights.cpu().numpy()
        for i, (src, dst) in enumerate(edge_list):
            G[src][dst]['weight'] = edge_weights[i]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Compute layout if not provided
    if node_pos is None:
        node_pos = nx.spring_layout(G)
    
    # Set node colors based on labels if available
    node_colors = None
    if node_labels is not None:
        node_colors = node_labels
    
    # Set edge colors based on weights if available
    edge_colors = None
    edge_widths = 1.0
    if edge_weights is not None:
        edge_colors = [w for _, _, w in G.edges(data='weight', default=1)]
        edge_widths = [1 + 2 * w for w in edge_colors]
    
    # Draw the graph
    nx.draw_networkx(G, pos=node_pos, with_labels=True, node_color=node_colors, cmap=plt.cm.tab10,
                     edge_color=edge_colors, width=edge_widths, edge_cmap=plt.cm.Blues,
                     node_size=300, font_size=10, font_color='black')
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return plt.gcf()

def visualize_embeddings(z, labels=None, perplexity=30, title=None, figsize=(10, 8), save_path=None):
    """Visualize node embeddings using t-SNE.
    
    Args:
        z: Node embeddings
        labels: Optional node labels for coloring
        perplexity: Perplexity parameter for t-SNE
        title: Optional plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure
    """
    # Convert to numpy if needed
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Apply t-SNE for dimensionality reduction
    z_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(z)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot embeddings
    if labels is not None:
        scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap=plt.cm.tab10, alpha=0.8)
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.8)
    
    if title:
        plt.title(title)
    
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return plt.gcf()

def visualize_curvature_distribution(edge_curvature, title=None, bins=30, figsize=(10, 6), save_path=None):
    """Visualize the distribution of edge curvatures.
    
    Args:
        edge_curvature: Dictionary mapping edges to curvature values
        title: Optional plot title
        bins: Number of histogram bins
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure
    """
    # Extract curvature values
    curvature_values = list(edge_curvature.values())
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot histogram
    plt.hist(curvature_values, bins=bins, alpha=0.8, color='steelblue', edgecolor='black')
    
    if title:
        plt.title(title)
    else:
        plt.title('Distribution of Edge Curvatures')
    
    plt.xlabel('Curvature')
    plt.ylabel('Frequency')
    
    # Add vertical line at curvature = 0
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add mean curvature line
    mean_curvature = np.mean(curvature_values)
    plt.axvline(x=mean_curvature, color='green', linestyle='-', alpha=0.7)
    plt.text(mean_curvature, plt.ylim()[1] * 0.9, f'Mean: {mean_curvature:.4f}', 
             horizontalalignment='center', verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return plt.gcf()

def visualize_loss_components(loss_history, title=None, figsize=(12, 8), save_path=None):
    """Visualize the loss components during training.
    
    Args:
        loss_history: Dictionary with loss components over epochs
        title: Optional plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot loss components
    epochs = list(range(1, len(loss_history['total_loss']) + 1))
    
    for loss_name, loss_values in loss_history.items():
        plt.plot(epochs, loss_values, label=loss_name)
    
    if title:
        plt.title(title)
    else:
        plt.title('Loss Components During Training')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return plt.gcf()

def visualize_graph_rewiring(original_edge_index, rewired_edge_index, node_pos=None, title=None, figsize=(15, 7), save_path=None):
    """Visualize the effect of graph rewiring by comparing original and rewired graphs.
    
    Args:
        original_edge_index: Original edge indices
        rewired_edge_index: Rewired edge indices
        node_pos: Optional node positions (if None, layout is computed)
        title: Optional plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure
    """
    # Convert to NetworkX graphs
    if isinstance(original_edge_index, torch.Tensor):
        original_edge_index = original_edge_index.cpu().numpy()
    
    if isinstance(rewired_edge_index, torch.Tensor):
        rewired_edge_index = rewired_edge_index.cpu().numpy()
    
    original_edge_list = list(zip(original_edge_index[0], original_edge_index[1]))
    rewired_edge_list = list(zip(rewired_edge_index[0], rewired_edge_index[1]))
    
    G_original = nx.Graph()
    G_original.add_edges_from(original_edge_list)
    
    G_rewired = nx.Graph()
    G_rewired.add_edges_from(rewired_edge_list)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Compute layout if not provided (use same layout for both graphs)
    if node_pos is None:
        node_pos = nx.spring_layout(G_original)
    
    # Draw original graph
    nx.draw_networkx(G_original, pos=node_pos, with_labels=True, 
                     node_color='skyblue', edge_color='gray',
                     node_size=300, font_size=10, font_color='black', ax=ax1)
    ax1.set_title('Original Graph')
    ax1.axis('off')
    
    # Draw rewired graph
    nx.draw_networkx(G_rewired, pos=node_pos, with_labels=True, 
                     node_color='lightgreen', edge_color='darkgreen',
                     node_size=300, font_size=10, font_color='black', ax=ax2)
    ax2.set_title('Rewired Graph')
    ax2.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig