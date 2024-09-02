import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(189, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def load_data(root_file, label):
    file = ROOT.TFile(root_file)
    tree = file.Get("events")
    shape_parameters = []
    for event in tree:
        shape_params = np.array(event._AugmentedCaloClusters_shapeParameters)
        if len(shape_params) >= 189:
            shape_parameters.append(shape_params[:189])
        else:
            print(f"Warning: Skipping event with less than 189 shape parameters: {len(shape_params)}")

    if len(shape_parameters) == 0:
        raise ValueError("No valid shape parameters found in the ROOT file.")

    shape_parameters = np.array(shape_parameters)
    shape_parameters = torch.tensor(shape_parameters, dtype=torch.float)
    labels = torch.tensor([label] * len(shape_parameters), dtype=torch.long)
    return shape_parameters, labels

def create_graph_data(shape_parameters, labels):
    data_list = []
    for i in range(len(shape_parameters)):
        x = shape_parameters[i].unsqueeze(0)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=labels[i].unsqueeze(0))
        data_list.append(data)
    return data_list

gamma_shape_params, gamma_labels = load_data("gamma_logE.root", 0)
pi0_shape_params, pi0_labels = load_data("pi0_logE.root", 1)

shape_parameters = torch.cat((gamma_shape_params, pi0_shape_params), dim=0)
labels = torch.cat((gamma_labels, pi0_labels), dim=0)

data_list = create_graph_data(shape_parameters, labels)
loader = DataLoader(data_list, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

all_labels = []
all_preds = []

model.train()
for epoch in range(100):
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    all_labels.append(data.y.cpu().numpy())
    all_preds.append(out[:, 1].detach().cpu().numpy())

all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)

fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.pdf')
plt.show()

torch.save(model.state_dict(), 'gnn_model.pth')

# model.load_state_dict(torch.load('gnn_model.pth'))

def classify_unknown_sample(unknown_root_file):
    unknown_shape_params, _ = load_data(unknown_root_file, -1)
    unknown_data_list = create_graph_data(unknown_shape_params, torch.tensor([-1] * len(unknown_shape_params)))

    model.eval()
    predictions = []
    with torch.no_grad():
        for data in unknown_data_list:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            predictions.append(pred.item())
            print(f"Prediction: {pred.item()}")

    return predictions

# classify_unknown_sample("unknown_sample.root")

