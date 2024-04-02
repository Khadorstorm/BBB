import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.nn import GCNConv
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops, Descriptors
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import *
# Load your dataset
def GCN(train_raw, test_raw, train_super_features, test_super_features):
    #scaler = StandardScaler()
    #normalized_features = scaler.fit_transform(X_train)
    graphs, labels = load_dataset(train_raw, train_super_features)
    dataset = MoleculeDataset(graphs, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(50):  # Number of epochs
        train(model, optimizer, loss_fn)
        print(f'Epoch {epoch + 1} Complete')

    # At this point, you can evaluate the model's performance on a test set in a similar manner.

    test_graphs, test_labels = load_dataset(test_raw, test_super_features)  # Adjust the path as needed
    test_dataset = MoleculeDataset(test_graphs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    true_labels, predictions = evaluate(test_loader)
    return predictions

df = pd.read_csv('~/PycharmProjects/BBB5/train/train_feature.csv')  # Ensure your CSV has columns 'SMILES' and 'label'
scaler = StandardScaler()
features = df.drop(['smi', 'label', 'id'], axis=1)  # Assuming these columns exist
normalized_features = scaler.fit_transform(features)
# Define a function to convert a molecule from SMILES to a graph
def mol_to_graph(smiles_string, super_node_features):
    mol = Chem.MolFromSmiles(smiles_string)
    if not mol:
        #print('kn')
        print(smiles_string)
        return None

    # Atom features: using atomic number and hybridization as example features
    atom_features = [[atom.GetAtomicNum(), atom.GetHybridization().real] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Bond features: using bond type as example features
    bond_features = []
    edge_index = [[], []]
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0].extend([start, end])
        edge_index[1].extend([end, start])
        bond_type = bond.GetBondTypeAsDouble()
        bond_features.extend([bond_type, bond_type])  # adding bond feature for both directions
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(bond_features, dtype=torch.float).view(-1, 1)

    # Calculate a global descriptor for the molecule, e.g., molecular weight
    mol_weight = Descriptors.MolWt(mol)
    super_node_feature = torch.tensor([mol_weight], dtype=torch.float).view(1, -1)
    # Assuming atom_features have 2 features, ensure super_node_feature also has 2 features
    # For example, by repeating the molecular weight or adding another global descriptor
    super_node_feature = torch.tensor( super_node_features, dtype=torch.float).view(1, -1)


    # Add super node to the graph
    # The supernode is connected to all other nodes
    num_nodes = x.size(0)
    super_node_index = torch.tensor([[num_nodes] * num_nodes, list(range(num_nodes))], dtype=torch.long)

    # Add super node features and edges to the graph
    x = torch.cat([x, super_node_feature], dim=0)
    edge_index = torch.cat([edge_index, super_node_index, super_node_index.flip(0)], dim=1)
    edge_attr = torch.cat([edge_attr, torch.zeros(2 * num_nodes, 1)], dim=0)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def load_dataset(df, super_features):
    graphs = []
    labels = []
    for index, row in df.iterrows():
        super_node_features = super_features[index]
        graph = mol_to_graph(row['smi'], super_node_features)
        if graph is not None:
            graphs.append(graph)
            labels.append(float(row['label']))
    return graphs, labels
# Define your dataset
class MoleculeDataset(Dataset):
    def __init__(self, graphs, labels):
        super(MoleculeDataset, self).__init__()
        self.graphs = graphs
        self.labels = labels

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx], self.labels[idx]

graphs, labels = load_dataset(df)
dataset = MoleculeDataset(graphs, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a simple GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 32)
        #self.conv3 = GCNConv(32, 64)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        #x = F.relu(self.conv2(x, edge_index))
        #x = F.dropout(x, training=self.training)

        #x = self.conv3(x, edge_index)

        x = torch_geometric.nn.global_mean_pool(x, data.batch)
        x = self.linear(x)
        return torch.sigmoid(x)

# Initialize the model, optimizer, and loss function
"""
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()
"""


# Train the model
def train(model,optimizer, loss_fn):
    model.train()
    for data, label in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, label.unsqueeze(1))
        loss.backward()
        optimizer.step()

for epoch in range(50):  # Number of epochs
    train()
    print(f'Epoch {epoch+1} Complete')

# At this point, you can evaluate the model's performance on a test set in a similar manner.

test_graphs, test_labels = load_dataset(df = pd.read_csv('test/test.csv'))  # Adjust the path as needed
test_dataset = MoleculeDataset(test_graphs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluation function
def evaluate(test_loader):
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for data, label in test_loader:
            out = model(data)
            predictions.extend(out.view(-1).cpu().numpy())
            true_labels.extend(label.cpu().numpy())
    return true_labels, predictions

# Evaluate model on the test set
true_labels, predictions = evaluate(test_loader)

# Calculate ROC curve and ROC area
fpr, tpr, _ = roc_curve(true_labels, predictions)
print(fpr,tpr)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()