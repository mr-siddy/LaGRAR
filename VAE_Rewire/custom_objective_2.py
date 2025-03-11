import torch

def neg_loss(z, adj_matrix, edge_index):
    neg_indices = []
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        Iv = torch.zeros_like(num_nodes, 1)
        torch.index_put_(Iv, torch.tensor(v), torch.tensor(1))
        Iu = torch.zeros_like(num_nodes, 1)
        torch.index_put_(Iu, torch.tensor(u), torch.tensor(1))
        adj_row_u = adj_matrix[u]
        adj_row_v = adj_matrix[v]
        sum_adj_u = torch.sum(adj_row_u)
        sum_adj_v = torch.sum(adj_row_v)
        term_1 = (torch.matmul(torch.matmul(A[u], z).t(), torch.matmul(Iv, z)) + torch.matmul(torch.matmul(A[v], z).t(), torch.matmul(Iu, z))) / (sum_adj_u + sum_adj_v)
        neg_indices.append(term_1.item())
    return neg_indices

def pos_loss(z, adj_matrix, edge_index):
    pos_indices = []
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])        
        adj_row_u = adj_matrix[u]
        adj_row_v = adj_matrix[v]
        sum_adj_u = torch.sum(adj_row_u)
        sum_adj_v = torch.sum(adj_row_v)
        term_2 = (torch.matmul(torch.matmul(A[u], z).t(), torch.matmul(Iv, z)) + torch.matmul(torch.matmul(A[v], z).t(), torch.matmul(Iu, z))) / torch.matmul(sum_adj_u, sum_adj_v)
        pos_indices.append(term_2.item())
    return pos_indices

