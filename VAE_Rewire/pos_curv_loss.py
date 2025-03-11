import torch
import torch_geometric
from collections import defaultdict

edge_index = data.edge_index
num_nodes = data.x.size(0)
adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
for i in range(edge_index.size(1)):
    u, v = int(edge_index[0, i]), int(edge_index[1, i])
    adj_matrix[u][v] = 1
    adj_matrix[v][u] = 1

adj_list = defaultdict(list)
for i in range(edge_index.size(1)):
    u, v = int(edge_index[0, i]), int(edge_index[1, i])
    adj_list[u].append(v)
    adj_list[v].append(u)

def pos_curv_loss(z, adj_list, edge_index):
    pos_indices = []
    num_nodes = len(adj_list.keys())
    for i in range(edge_index.size(1)):
        alpha, beta, gamma = 0
        i, j = int(edge_index[0, i]), int(edge_index[1, i])
        d_u, d_v = len(adj_list[u]), len(adj_list[v])
        for u in range(adj_list[i]):
            alpha += torch.matmul(z[u], z[j]) / torch.matmul(torch.norm(z[u]), torch.norm(z[j]))
            for v in range(adj_list[j]):
                if v != i:
                    alpha += torch.matmul(z[v], z[i]) / torch.matmul(torch.norm(z[v]), torch.norm(z[i]))
                    alpha = alpha/ max(d_u, d_v)
                if j != u:
                    beta += torch.matmul(z[u], z[v]) / torch.matmul(torch.norm(z[u]), torch.norm(z[v]))
                    beta = beta/ max(d_u, d_v)
                for w in range(adj_list[v]):
                    if w != j and w != i:
                        gamma += torch.matmul(z[w], z[u]) / torch.matmul(torch.norm(z[w]), torch.norm(z[u]))
                        gamma = gamma/ max(d_u, d_v)
        pos_indices.append(alpha+beta+gamma)
    return pos_indices