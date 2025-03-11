# modules/curvature.py
import networkx as nx
import numpy as np
from scipy.optimize import linprog

def compute_ollivier_ricci_curvature(G, p=0.5):
    """Compute the Ollivier-Ricci curvature for all edges in the graph.
    
    Args:
        G: NetworkX graph
        p: Probability for lazy random walk (default: 0.5)
        
    Returns:
        Dictionary mapping edges to their Ollivier-Ricci curvature
    """
    curvatures = {}
    
    for u, v in G.edges():
        # Compute the random walk probability measures
        mu_x = compute_random_walk_measure(G, u, p)
        mu_y = compute_random_walk_measure(G, v, p)
        
        # Compute the Wasserstein distance using linear programming
        W1 = wasserstein_distance(G, mu_x, mu_y)
        
        # Compute the edge distance (typically 1 for unweighted graphs)
        edge_dist = G[u][v].get('weight', 1.0)
        
        # Compute the Ollivier-Ricci curvature
        kappa = 1 - (W1 / edge_dist)
        curvatures[(u, v)] = kappa
        curvatures[(v, u)] = kappa  # Undirected graph, same curvature
    
    return curvatures

def compute_random_walk_measure(G, node, p=0.5):
    """Compute the random walk probability measure for a node.
    
    Args:
        G: NetworkX graph
        node: Source node
        p: Probability to stay at the current node (lazy random walk)
        
    Returns:
        Dictionary mapping nodes to their probability mass
    """
    measure = {node: p}  # Probability p to stay at current node
    
    # Distribute remaining probability (1-p) to neighbors
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return measure
    
    prob_per_neighbor = (1.0 - p) / len(neighbors)
    
    for neighbor in neighbors:
        measure[neighbor] = prob_per_neighbor
    
    return measure

def wasserstein_distance(G, mu_x, mu_y):
    """Compute the 1-Wasserstein distance between two probability measures.
    
    Args:
        G: NetworkX graph
        mu_x: First probability measure (dict mapping nodes to probabilities)
        mu_y: Second probability measure (dict mapping nodes to probabilities)
        
    Returns:
        1-Wasserstein distance (float)
    """
    # Get all nodes with non-zero probability
    supp_x = set(mu_x.keys())
    supp_y = set(mu_y.keys())
    nodes = list(supp_x | supp_y)
    n = len(nodes)
    
    # Build node-to-index mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Prepare the cost matrix (using shortest path distances)
    cost_matrix = np.zeros((n, n))
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if i == j:
                cost_matrix[i, j] = 0
            else:
                try:
                    cost_matrix[i, j] = nx.shortest_path_length(G, source, target, weight='weight')
                except nx.NetworkXNoPath:
                    cost_matrix[i, j] = float('inf')
    
    # Flatten the cost matrix for linear programming
    costs = cost_matrix.flatten()
    
    # Prepare the constraints for the linear program
    # Row constraints: sum_j pi_ij = mu_x(i)
    row_constraints = []
    for i in range(n):
        constraint = np.zeros(n * n)
        constraint[i * n:(i + 1) * n] = 1
        row_constraints.append(constraint)
    
    # Column constraints: sum_i pi_ij = mu_y(j)
    col_constraints = []
    for j in range(n):
        constraint = np.zeros(n * n)
        constraint[j::n] = 1
        col_constraints.append(constraint)
    
    # Combine constraints
    A_eq = np.vstack(row_constraints + col_constraints)
    
    # Right-hand side of constraints
    b_eq_row = [mu_x.get(nodes[i], 0) for i in range(n)]
    b_eq_col = [mu_y.get(nodes[j], 0) for j in range(n)]
    b_eq = np.array(b_eq_row + b_eq_col)
    
    # Solve the linear program
    result = linprog(costs, A_eq=A_eq, b_eq=b_eq, method='highs')
    
    # Return the