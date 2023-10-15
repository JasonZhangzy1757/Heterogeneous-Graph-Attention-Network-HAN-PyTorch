import torch
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F


class RGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim, num_classes):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(num_nodes, hidden_dim, num_relations, num_bases=30)
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations, num_bases=30)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)