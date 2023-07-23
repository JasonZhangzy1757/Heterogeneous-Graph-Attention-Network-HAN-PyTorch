import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeAttentionLayer(nn.Module):
    """
    Adapted from Diego999/pyGAT
    """
    def __init__(self, in_feature_dim, out_feature_dim, dropout, alpha):
        super(NodeAttentionLayer, self).__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.dropout = dropout
        # The paper didn't specify but the author used the default 0.2 in tensorflow.
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.weight = nn.Parameter(torch.empty(size=(self.in_feature_dim, self.out_feature_dim)))
        self.attention_coef = nn.Parameter(torch.empty(size=(self.out_feature_dim * 2, 1)))
        # Initiate with the recommended value of the leaky relu with a slope of 0.2.
        nn.init.xavier_uniform_(self.weight, gain=1.387)
        nn.init.xavier_uniform_(self.attention_coef, gain=1.387)

    def forward(self, x, adj):
        Wh = torch.mm(x, self.weight)      # Wh: (N, out_feature_dim)
        e = self._prepare_attention(Wh)    # e: (N, N) So this could be seen as an interaction matrix

        infneg_vector = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, infneg_vector)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # h_prime: (N, out_feature_dim)

        return F.elu(h_prime)

    def _prepare_attention(self, Wh):
        Wh1 = torch.matmul(Wh, self.attention_coef[:self.out_feature_dim, :])  # Wh1 & Wh2: (N, 1)
        Wh2 = torch.matmul(Wh, self.attention_coef[self.out_feature_dim:, :])
        e = Wh1 + Wh2.T  # Broadcast add

        return self.leakyrelu(e)


class SemanticAttentionLayer(nn.Module):
    def __init__(self, in_feature_dim, q_vector):
        super(SemanticAttentionLayer, self).__init__()
        self.weight = nn.Parameter(torch.empty(size=(in_feature_dim, q_vector)))
        self.bias = nn.Parameter(torch.empty(size=(1, q_vector)))
        self.q = nn.Parameter(torch.empty(size=(q_vector, 1)))

        # Similarly, the recommended gain value for tanh
        nn.init.xavier_uniform_(self.weight, gain=1.667)
        nn.init.xavier_uniform_(self.bias, gain=1.667)
        nn.init.xavier_uniform_(self.q, gain=1.667)

    def forward(self, z):
        Wh = torch.matmul(z, self.weight) + self.bias    # z: (N, M, hidden_dim * num_classes)
        Wh = F.tanh(Wh)                 # Wh: (N, M, q_vector)
        w = torch.matmul(Wh, self.q)    # w: (N, M, 1)
        w = w.mean(0)                   # w: (M, 1)
        beta = F.softmax(w, dim=1)
        beta = beta.expand((z.shape[0],) + beta.shape)    # beta: (N, M, 1)

        return (beta * z).sum(1)       # (N, hidden_dim * num_classes)
