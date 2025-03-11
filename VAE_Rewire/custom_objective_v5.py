import torch

adj_list = defaultdict(list)
for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        adj_list[u].append(v)
        adj_list[v].append(u)