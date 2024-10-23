import h5py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

# GPU setup: Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Data loading function
def load_data(h5_file, label):
    with h5py.File(h5_file, 'r') as f:
        cells = f['cells'][:]  # cell shape: (num_events, num_cells, 4)
        clusters = f['clusters'][:]  # clusters shape: (num_events, 4)

    data_list = []
    for i in range(len(clusters)):
        cluster_features = clusters[i]  # Each cluster's energy and position
        cell_features = cells[i]  # Each cell's energy and position

        # Construct graph using cell's position info (x, y, z) as edge calculation base
        edge_index = torch.tensor([(i, j) for i in range(len(cell_features)) for j in range(len(cell_features))],
                                  dtype=torch.long).t().contiguous()

        x = torch.tensor(cell_features, dtype=torch.float)  # Cell features
        y = torch.tensor([label], dtype=torch.float)  # Signal or background label

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list

# Load signal and background data
#signal_data = load_data('../../gamma_5000_max_cluster.h5', label=1)
#background_data = load_data('../../pi0_5000_max_cluster.h5', label=0)
signal_data = load_data('../../gamma_5000_max_cluster.h5', label=1)[:100]  # 仅加载前1000个数据进行调试
background_data = load_data('../../pi0_5000_max_cluster.h5', label=0)[:100]  # 仅加载前1000个数据


# Combine signal and background data
data_list = signal_data + background_data
loader = DataLoader(data_list, batch_size=4, shuffle=True)
##loader = DataLoader(data_list, batch_size=32, shuffle=True)

# Define GNN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(4, 64)  # Input: 4 features, Output: 64 features
        self.conv2 = GCNConv(64, 32)  # Intermediate: 64 features, Output: 32 features
        self.fc = torch.nn.Linear(32, 1)  # Fully connected layer for classification

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Aggregate information from all nodes
        x = self.fc(x)
        return torch.sigmoid(x)  # Output probability

# Initialize model and optimizer
model = GCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()

# Training function
def train(model, loader):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.unsqueeze(1).to(device))  # Compute loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        # Show progress every 50 batches
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')

    return total_loss / len(loader.dataset)

# Testing function
def test(model, loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            preds.append(out.cpu().numpy())
            labels.append(data.y.numpy())
    return np.concatenate(preds), np.concatenate(labels)

# Training the model
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    loss = train(model, loader)
    print(f'Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}')

# Get predictions
preds, labels = test(model, loader)

# Plot ROC curve
fpr, tpr, _ = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Save the ROC curve image
plt.savefig('roc_curve.png')
print('ROC curve saved as roc_curve.png')

# Show the plot
plt.show()

