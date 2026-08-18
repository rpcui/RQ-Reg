import torch
import torch.nn as nn

class FeatureEmbedding(nn.Module):
    def __init__(self, data_config, model_config):
        super(FeatureEmbedding, self).__init__()
        dim = model_config['embedding_dim']
        self.seq_feature_index = data_config['seq_feature_index']

        self.sparse_embedding = nn.ModuleList([
            nn.Embedding(size, dim)
            for size in data_config['embedding_size']['sparse']
        ])
        self.seq_embedding = nn.ModuleList([
            nn.Embedding(size, dim)
            for size in data_config['embedding_size']['seq']
        ])
        self.dense_embedding = nn.ModuleList([
            nn.Embedding(size, dim)
            for size in data_config['embedding_size']['dense']
        ])

    def forward(self, x_sparse, x_seq, dense_feature, dense_coef):
        sparse_emb = []
        seq_emb = []
        dense_emb = []

        for idx, embedding in enumerate(self.sparse_embedding):
            sparse_emb.append(embedding(x_sparse[:, idx]))  # [B, dim]

        for idx, embedding in enumerate(self.seq_embedding):
            emb = embedding(x_seq[:, self.seq_feature_index[idx]: self.seq_feature_index[idx + 1]])
            emb = torch.sum(emb, dim=1)
            seq_emb.append(emb)

        for idx, embedding in enumerate(self.dense_embedding):
            emb = embedding(dense_feature[:, 2 * idx: 2 * (idx + 1)])
            coef = dense_coef[:, 2 * idx: 2 * (idx + 1)][..., None]
            dense_emb.append(torch.sum(emb * coef, dim=1))

        return torch.cat(sparse_emb + seq_emb + dense_emb, dim=1)

class TokenEmbedding(nn.Module):
    def __init__(self, data_config, model_config):
        super(TokenEmbedding, self).__init__()

        self.embedding = nn.Embedding(data_config['vocab_size'], model_config['d_model'])

    def forward(self, x):
        return self.embedding(x)

    def mix_forward(self, x):
        embedding = torch.matmul(x, self.embedding.weight)
        return embedding