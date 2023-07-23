import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import NodeAttentionLayer, SemanticAttentionLayer


class HAN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, dropout, num_heads, alpha, q_vector):
        super(HAN, self).__init__()
        self.dropout = dropout
        self.q_vector = q_vector

        self.attentions = [NodeAttentionLayer(feature_dim, hidden_dim, self.dropout, alpha) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.semantic_attention = SemanticAttentionLayer(hidden_dim * num_heads, q_vector)
        self.out_layer = nn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, x, meta_path_list):
        semantic_embeddings = []
        for meta_path_adj in meta_path_list:
            x = F.dropout(x, self.dropout, training=self.training)
            Z = torch.cat([attention(x, meta_path_adj) for attention in self.attentions], dim=1)
            Z = F.dropout(Z, self.dropout, training=self.training)
            semantic_embeddings.append(Z)

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        final_embedding = self.semantic_attention(semantic_embeddings)

        return self.out_layer(final_embedding)

