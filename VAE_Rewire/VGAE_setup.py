import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim * 2)  # Multiply by 2 for mean and variance

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))  # Assuming binary adjacency matrix
        return z

# Define the VAE
class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        # Encode
        encoded = self.encoder(x, edge_index)
        mu, log_var = encoded.chunk(2, dim=-1)  # Split into mean and log variance
        z = self.reparameterize(mu, log_var)

        # Decode
        decoded = self.decoder(z)
        return decoded, mu, log_var

# Reconstruction loss and KL divergence
def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# Example usage - Training loop
def train_model(model, data_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            recon, mu, log_var = model(data.x, data.edge_index)
            loss = loss_function(recon, data.x, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_loader.dataset)}")

# Example dataset creation and training
# Assuming you have a list of adjacency matrices or graphs
# Create PyTorch Geometric Data objects and put them in a DataLoader
# For example:
adj_matrices = [...]  # List of adjacency matrices
data_list = [Data(adj=torch.Tensor(adj).to_sparse()) for adj in adj_matrices]
data_loader = DataLoader(data_list, batch_size=32, shuffle=True)

# Define model, optimizer, and train the VAE
input_dim = ...  # Define input dimensions based on your data
hidden_dim = 64
latent_dim = 16
model = GraphVAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, data_loader, optimizer, num_epochs=10)
