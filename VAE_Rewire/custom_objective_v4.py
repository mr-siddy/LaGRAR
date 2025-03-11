import torch

def neg_loss(z, adj_matrix, edge_index):
    neg_indices = []
    num_nodes = adj_matrix.size(0)
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        Iv, Iu = torch.zeros(num_nodes), torch.zeros(num_nodes)
        Iv[v] = 1
        Iu[u] = 1
        Iv = Iv.unsqueeze(1)
        Iu = Iu.unsqueeze(1)

        adj_row_u = adj_matrix[u]
        adj_row_v = adj_matrix[v]

        adj_row_u = adj_row_u.unsqueeze(1)
        adj_row_v = adj_row_v.unsqueeze(1)
        sum_adj_u = torch.sum(adj_row_u)
        sum_adj_v = torch.sum(adj_row_v)

        term_1 = torch.einsum('ij,ij->', adj_row_u * z, Iv * z) + torch.einsum('ij,ij->', adj_row_v * z, Iu * z)
        term_1 /= (sum_adj_u + sum_adj_v)

        neg_indices.append(term_1.item())

    return neg_indices


def pos_loss(z, adj_matrix, edge_index):
    pos_indices = []
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        adj_row_u = adj_matrix[u]
        adj_row_v = adj_matrix[v]
        adj_row_u = adj_row_u.unsqueeze(1)
        adj_row_v = adj_row_v.unsqueeze(1)
        sum_adj_u = torch.sum(adj_row_u)
        sum_adj_v = torch.sum(adj_row_v)
        Iv, Iu = torch.zeros(num_nodes), torch.zeros(num_nodes)
        Iv[v] = 1
        Iu[u] = 1
        Iv = Iv.unsqueeze(1)
        Iu = Iu.unsqueeze(1)
        # term_2 = (torch.matmul((adj_row_u * z).t(), (adj_row_v * z))) / (sum_adj_u * sum_adj_v)
        #term_2 = torch.einsum('ij,ij->', adj_row_u * z, adj_row_v * z) / (sum_adj_u * sum_adj_v)
        term_3 = (torch.einsum('ij,ij->', adj_row_u * z, adj_row_v * z)) 
        term_3 /= (torch.einsum('ij, ij->', Iu * z, adj_row_u * z) * torch.einsum('ij, ij->', Iv * z, adj_row_v * z))
        pos_indices.append(term_3.item())
    return pos_indices