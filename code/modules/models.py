import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.maxpool = nn.MaxPool2d(2) 
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.head(x)


# mlp taking array of normalized fourier descriptors
class MLP(nn.Module):
    def __init__(self, num_classes, fourier_order):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
        nn.Linear(fourier_order*4, 512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,num_classes))

    def forward(self, x):
        x = self.flatten(x)
        out = self.mlp(x)
        return out


# pytorch-geometric GCN
class GNN(nn.Module):
  def __init__(self, num_classes, fourier_order, width_mult, edge_attr_dim, has_skip, is_deep):
    super(GNN, self).__init__()
    self.edge_attr_dim = edge_attr_dim
    self.has_skip = has_skip
    self.is_deep = is_deep

    self.embedding = nn.Sequential(
                                    nn.Linear(fourier_order*4, int(width_mult*512)),
                                    nn.ReLU(),
                                    nn.Linear(int(width_mult*512), int(width_mult*512)),
                                    nn.ReLU(),
                                )
    self.edge_proj = nn.Sequential(
                                nn.Linear(edge_attr_dim, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1),
                                nn.Sigmoid()
                            )
    self.relu = nn.ReLU()
    self.conv1 = GCNConv(int(width_mult*512), int(width_mult*1024))
    self.conv2 = GCNConv(int(width_mult*1024), int(width_mult*1024))
    self.conv3= GCNConv(int(width_mult*1024), int(width_mult*1024))
    self.conv4 = GCNConv(int(width_mult*1024), int(width_mult*1024))
    self.conv5 = GCNConv(int(width_mult*1024), int(width_mult*1024))
    self.conv6= GCNConv(int(width_mult*1024), int(width_mult*1024))
    self.fc1 = nn.Linear(int(width_mult*1024), int(width_mult*2048))
    self.head = nn.Linear(int(width_mult*2048), num_classes)

  def forward(self, data):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

    if self.edge_attr_dim > 1:
      edge_attr = edge_attr.squeeze(dim=2)
    edge_weight = self.edge_proj(edge_attr)
    x = self.embedding(x)
    x = self.conv1(x, edge_index, edge_weight)
    skip = self.relu(x)
    x = self.conv2(skip, edge_index, edge_weight)
    x = self.relu(x)
    x = self.conv3(x, edge_index, edge_weight)
    if self.has_skip:
        skip = self.relu(x) + skip
    else:
        skip = self.relu(x)
    if self.is_deep:
        x = self.conv4(skip, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv6(x, edge_index, edge_weight)
        if self.has_skip:
            skip = self.relu(x) + skip
        else:
            skip = self.relu(x)
    x = global_mean_pool(skip, batch)
    x = self.fc1(x)
    x = self.relu(x)
    return self.head(x)