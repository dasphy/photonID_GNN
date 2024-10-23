import os
import uproot
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch_geometric

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def extract_clusters_and_cells(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    cluster_data = []
    labels = []
    
    with uproot.open(file_path) as file:
        tree = file["events"]
        for entry in tree.iterate(
            [
                "CorrectedCaloClusters.energy", 
                "CorrectedCaloClusters.position.x", 
                "CorrectedCaloClusters.position.y", 
                "CorrectedCaloClusters.position.z", 
                "PositionedCaloClusterCells.energy", 
                "PositionedCaloClusterCells.position.x", 
                "PositionedCaloClusterCells.position.y", 
                "PositionedCaloClusterCells.position.z"
            ], 
            library="pd"
        ):
            cluster_energies = entry["CorrectedCaloClusters.energy"].to_numpy()
            if len(cluster_energies) == 0:
                continue
            
            max_energy_index = np.argmax(cluster_energies)
            cluster_position = np.array([
                entry["CorrectedCaloClusters.position.x"].to_numpy()[max_energy_index],
                entry["CorrectedCaloClusters.position.y"].to_numpy()[max_energy_index],
                entry["CorrectedCaloClusters.position.z"].to_numpy()[max_energy_index]
            ])
            
            cell_energies = entry["PositionedCaloClusterCells.energy"].to_numpy()
            cell_positions = np.column_stack((
                entry["PositionedCaloClusterCells.position.x"].to_numpy(),
                entry["PositionedCaloClusterCells.position.y"].to_numpy(),
                entry["PositionedCaloClusterCells.position.z"].to_numpy()
            ))

            if isinstance(cell_energies[0], list):
                cell_energies = np.concatenate(cell_energies)
            else:
                cell_energies = np.array(cell_energies, dtype=np.float32)

            labels.append(1 if "signal" in file_path else 0)

            cluster_data.append((cell_energies, cell_positions, cluster_position))
    
    return cluster_data, labels

def create_graph_data(cluster_data):
    graphs = []
    for cell_energies, cell_positions, cluster_position in cluster_data:
        num_cells = len(cell_energies)
        if num_cells == 0:
            continue
        
        node_features = torch.tensor(cell_energies, dtype=torch.float).view(-1, 1)
        
        edge_index = []
        for i in range(num_cells):
            for j in range(num_cells):
                if i != j:
                #if i != j and np.linalg.norm(cell_positions[i] - cell_positions[j]) < 0.1:
                    edge_index.append((i, j))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        graph = Data(x=node_features, edge_index=edge_index)
        graphs.append(graph)
    
    return graphs

def main(signal_file_path, background_file_path):
    signal_data, signal_labels = extract_clusters_and_cells(signal_file_path)
    background_data, background_labels = extract_clusters_and_cells(background_file_path)
    
    all_data = signal_data + background_data
    all_labels = signal_labels + background_labels
    
    graphs = create_graph_data(all_data)
    
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(graphs, all_labels, test_size=0.2, random_state=42)

    model = GNN(in_channels=1, hidden_channels=16, out_channels=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(100):
        total_loss = 0
        for graph, label in zip(train_graphs, train_labels):
            optimizer.zero_grad()
            out = model(graph)
            label_tensor = torch.tensor([label], dtype=torch.float)
            loss = criterion(out, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_graphs)}')

    model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for graph, label in zip(test_graphs, test_labels):
            out = model(graph)
            y_true.append(label)
            y_scores.append(torch.sigmoid(out).item())
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
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
    plt.savefig("roc_curve.png")
    plt.show()

if __name__ == "__main__":
    signal_file_path = "./gamma.root"
    background_file_path = "./pi0.root"
    main(signal_file_path, background_file_path)

