import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.datasets import WebKB, Planetoid, WikipediaNetwork, Actor
from torch_geometric.nn import GCNConv, VGAE
#from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torch import tensor
from collections import defaultdict, deque

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset =  WebKB(root="/home/siddy/META/data", name="texas")
data = dataset[0].to(device)

edge_index= data.edge_index
num_nodes = data.x.size(0)
adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
for i in range(edge_index.size(1)):
    u, v = int(edge_index[0, i]), int(edge_index[1, i])
    adj_matrix[u, v] = 1
    adj_matrix[v, u] = 1

adj_list = defaultdict(list)
for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        adj_list[u].append(v)
        adj_list[v].append(u)


# below code provides functions that mimics the above code using the einsum operations in the pytorch bbut provides a scaler value instead of a vector
def pos_loss(z, adj_list, edge_index):
    pos_indices = []
    num_nodes = len(adj_list.keys())

    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        d_u = len(adj_list[u])
        d_v = len(adj_list[v])
        u_neighbors_norm = torch.stack([z[node] / torch.norm(z[node]) for node in adj_list[u]])
        v_neighbors_norm = torch.stack([z[node] / torch.norm(z[node]) for node in adj_list[v]])

        term_u = torch.einsum('ij,j->', u_neighbors_norm, z[v])
        u_ = term_u / d_u
        term_v = torch.einsum('ij, j->', v_neighbors_norm, z[u])
        v_ = term_v / d_v
        pos_indices.append(u_ + v_)

    return pos_indices

def neg_loss(z, adj_list, edge_index):
    neg_indices = []
    num_nodes = len(adj_list.keys())

    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        d_u = len(adj_list[u])
        d_v = len(adj_list[v])

        u_neighbors_norm = torch.stack([z[node] / torch.norm(z[node]) for node in adj_list[u]])
        v_neighbors_norm = torch.stack([z[node] / torch.norm(z[node]) for node in adj_list[v]])

        for u in u_neighbors_norm:
          for v in v_neighbors_norm:
            z_u_dot_z_v = torch.matmul(u, v.T)
        term_ = torch.sum(z_u_dot_z_v) / (d_u * d_v)
        neg_indices.append(term_)

    return neg_indices

def custom_objective_2(pos_loss, neg_loss):
  wandb.log({"pos_indices":pos_loss})
  wandb.log({"neg_indices": neg_loss})
  return (pos_loss+2) / 2*(neg_loss+1)

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(0.5)
        deg_inv_sqrt[deg_inv_sqrt==float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # propagating messages
        out = self.propagate(edge_index, x=x, norm=norm)
        out += self.bias
        return torch.squeeze(out)

    def message(self, x_j, norm):
        return norm.view(-1,1) *x_j

class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.fc = Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(self.conv(x, edge_index))
        return x

# Encoder
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self,x ,edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

out_channels= dataset.num_classes
num_features = dataset.num_features
epochs=300
num_nodes_train = data.x.size(0)

model = VGAE(GCNEncoder(num_features, out_channels))
gcn_net = GCNNet(in_channels=data.x.size(1), hidden_channels=64, out_channels=dataset.num_classes)
gcn_net = gcn_net.to(device)
model = model.to(device)
x = data.x.to(device)
optimizer1 = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)
optimizer2 = torch.optim.Adam(gcn_net.parameters(), lr=0.01, weight_decay=1e-3)

#Generator function for k-fold dataset
def cross_validation_with_test_set(dataset, folds):
    cross_val_dataset = []
    for fold, (train_idx, test_idx) in enumerate(zip(*kfold(dataset, folds))):
        data = Data(x=dataset.data.x, edge_index=dataset.data.edge_index, y=dataset.data.y)
        train_mask = torch.zeros([dataset.data.x.shape[0]], dtype=torch.bool)
        test_mask = torch.zeros([dataset.data.x.shape[0]], dtype=torch.bool)
        train_mask[train_idx]=True
        test_mask[test_idx] = True
        data.train_mask = train_mask
        data.test_mask = test_mask
        cross_val_dataset.append(data)
    return cross_val_dataset

# implemented train-test k-fold
def kfold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=1610)
    train_indices, test_indices = [], []
    for idx, (train_idx, test_idx) in enumerate(skf.split(dataset.data.x, dataset.data.y)):
        train_idx, test_idx = (train_idx, test_idx)
        train_indices.append(torch.from_numpy(train_idx).to(torch.long))
        test_indices.append(torch.from_numpy(test_idx).to(torch.long))
    return train_indices, test_indices

def train(train_data, adj_list):
    model.train()
    gcn_net.train()
    optimizer1.zero_grad()

    # neg_edge_index = negative_sampling(
    #     edge_index= data.edge_index,
    #     num_nodes= data.x.size(0),
    #     num_neg_samples= data.edge_index.size(1)
    # )
    # neg_indices = neg_index(num_nodes_train, train_data.edge_index)
    # pos_indices = pos_index(num_nodes_train, train_data.edge_index)
    z = model.encode(x, data.edge_index)
    print(f"latent space shape: {z.shape}")
    #adj = torch.sigmoid(torch.matmul(z, z.t()))
    #print(f"adj matrix shape: {adj.shape}")
    loss = model.recon_loss(z, train_data.edge_index)
    wandb.log({"recon_loss":loss.item()})
    kl_loss = (1 / data.num_nodes) * model.kl_loss()  # new line
    wandb.log({"kl_loss":kl_loss.item()})
    loss += kl_loss
    #adj_binary = (adj > 0.5).float()
    #edge_list = adj.nonzero(as_tuple=False)
    #edge_list = torch.permute(torch.tensor(edge_list, dtype=torch.long), (1,0)).to(device)
    neg_indices = neg_loss(z, adj_list, train_data.edge_index)
    pos_indices = pos_loss(z, adj_list, train_data.edge_index)
    objective_loss = 0.0
    for i in range(len(neg_indices)):
        sigma_1 = torch.tensor(neg_indices[i], requires_grad=True)
        sigma_2 = torch.tensor(pos_indices[i], requires_grad=True)
        objective = custom_objective_2(sigma_1, sigma_2)
        objective_loss += objective
    #objective_loss /= len(edge_index[0])
    wandb.log({"obj_loss":objective_loss.item()})
    loss += objective_loss
    optimizer2.zero_grad()
    out = gcn_net(data.x, data.edge_index)
    out_2 = gcn_net(data.x, data.edge_index)
    nc_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    nc_loss_2 = F.cross_entropy(out_2[data.train_mask], data.y[data.train_mask])
    wandb.log({"nc_loss":nc_loss.item()})
    loss += nc_loss
    loss.backward()
    optimizer1.step()
    nc_loss_2.backward()
    optimizer2.step()
    return nc_loss_2, loss


def test(test_data):
    model.eval()
    gcn_net.eval()
    with torch.no_grad():
        # test_neg_edge_index = negative_sampling(
        #     edge_index= test_data.edge_index,
        #     num_nodes= test_data.x.size(0),
        #     num_neg_samples=test_data.edge_index.size(1)
        # )
        # #z = model.encode(x, test_data.edge_index)
        out = gcn_net(data.x, data.edge_index).argmax(dim=-1)
        accs=[]
        for mask in [data.train_mask, data.test_mask]:
            accs.append(int((out[mask] == data.y[mask]).sum())/ int(mask.sum()))
        return accs

    #return model.test(z, test_data.edge_index, test_neg_edge_index)
folds= 5
epochs=300
cross_val_dataset = cross_validation_with_test_set(dataset, folds)
loader = DataLoader(cross_val_dataset, batch_size=1, shuffle=False)

import time
epochs=500
best_val_acc = final_test_acc = 0
times = []
train_losses, val_accs, test_accs = [], [], []
for data in loader:
  data = data.to(device)
  for epoch in range(1, epochs+1):
      start = time.time()
      _, loss= train(data, adj_list)
      wandb.log({"total_loss":loss})
      train_losses.append(loss)
      train_acc, test_acc = test(data)
      val_accs.append(train_acc)
      test_accs.append(test_acc)
      print(f"Epoch:{epoch}, Loss:{loss}, Train:{train_acc}, Test:{test_acc}")
      wandb.log({"train_acc":train_acc})
      wandb.log({"test_acc":test_acc})

loss, acc = tensor(train_losses), tensor(test_accs)
loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
print(loss.shape, acc.shape)

loss, argmin = loss.min(dim=1)
acc = acc[torch.arange(folds, dtype=torch.long), argmin]
print(loss, acc)
loss_mean = loss.mean().item()
acc_mean = acc.mean().item()
acc_std = acc.std().item()

print(f"train_loss:{loss_mean:.4f}, Test Acuracy: {acc_mean:.3f}" f"+- {acc_std:.3f}")
wandb.log({"mean_train_loss": loss_mean})
wandb.log({"mean_test_acc": acc_mean})
wandb.log({"test_std": acc_std})

run_vgae_rewire.finish()