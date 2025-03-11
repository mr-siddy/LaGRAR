import torch
from collections import defaultdict, deque

def neg_index(num_nodes, edge_index):
    adj_list = defaultdict(list)
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        adj_list[u].append(v)
        adj_list[v].append(u)

    neg_indices = []
    for i in range(num_nodes):
        visited = set()
        queue = deque([(i, 0)])
        visited.add(i)
        count_two_away = 0
        while queue:
            node, distance = queue.popleft()
            if distance == 2:
                count_two_away += 1
                continue
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        sum_adj_i = len(adj_list[i])
        neg_index = count_two_away / sum_adj_i if sum_adj_i > 0 else 0
        neg_indices.append(neg_index)

    return neg_indices


def pos_index(num_nodes, edge_index):
    adj_list = defaultdict(list)
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        adj_list[u].append(v)
        adj_list[v].append(u)

    pos_indices = []
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        common_neighbors = 0
        visited = set()
        queue = deque(adj_list[u])
        visited.add(u)
        
        while queue:
            node = queue.popleft()
            if node in adj_list[v]:
                common_neighbors += 1
            visited.add(node)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        sum_adj_u = len(adj_list[u])
        sum_adj_v = len(adj_list[v])
        pos_index = common_neighbors / (sum_adj_u + sum_adj_v)
        pos_indices.append(pos_index)

    return pos_indices
