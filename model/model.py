import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from model.embedding import FeatureEmbedding, TokenEmbedding
from model.lstm import LSTMModule

class RQGenModel(nn.Module):
    def __init__(self, data_config, model_config):
        super(RQGenModel, self).__init__()

        self.d_model = model_config['d_model']
        input_dim = model_config['input_dim']
        self.num_layers = model_config['num_layers']

        self.num_step = data_config['codebook_layer']
        self.register_buffer('vocabulary', torch.tensor(data_config['vocabulary'], dtype=torch.float))

        self.feature_embedding = FeatureEmbedding(data_config, model_config)
        self.token_embedding = TokenEmbedding(data_config, model_config)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * self.d_model, bias=True),
            nn.ReLU(),
            nn.Linear(2 * self.d_model, 2 * self.d_model * self.num_layers, bias=True)
        )
        self.lstm = LSTMModule(model_config)
        self.predictor = nn.Linear(self.d_model, data_config['vocab_size'])
        self.regressor = nn.Sequential(
            nn.Linear(data_config['vocab_size'], self.d_model, bias=True),
            nn.ReLU(),
            nn.Linear(self.d_model, 1, bias=True)
        )
        self.align_proj = nn.Linear(data_config['vocab_size'], self.d_model, bias=False)
        self.error_regressor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            nn.ReLU(),
            nn.Linear(self.d_model, 1, bias=True)
        )

    def forward(self, x_sparse, x_seq, dense_feature, dense_coef, x_token, sampling_ratio):
        x = self.feature_embedding(x_sparse, x_seq, dense_feature, dense_coef)
        x = self.encoder(x)
        x = x.reshape(x.shape[0], self.num_layers, -1).permute(1, 0, 2)
        hidden_states = torch.split(x, [self.d_model, self.d_model], dim=2)  # (num_layers, B, d_model)
        hidden_states = (hidden_states[0].contiguous(), hidden_states[1].contiguous())

        emb = self.token_embedding(x_token)

        align_feat1 = []
        align_feat2 = []

        for t in range(self.num_step):
            if t > 0 and random.random() < sampling_ratio:
                emb_input = mix_emb
            else:
                emb_input = emb[:, [t]]
            output, (h_t, c_t) = self.lstm(emb_input, hidden_states)
            hidden_states = (h_t, c_t)
            logit = self.predictor(output)
            result = self.regressor(logit)
            feat = self.align_proj(logit)
            if t == 0:
                logits = [logit]
                results = [result]
            else:
                logits.append(logit)
                results.append(result)
            mix_emb = self.token_embedding.mix_forward(F.softmax(logit, dim=-1))
            align_feat1.append(mix_emb)  # (B, 1, d_model)
            align_feat2.append(feat)

        # add regression for RQ error
        err = self.error_regressor(mix_emb)  # (B, 1, 1)
        results.append(err)

        logits = torch.cat(logits, dim=1)
        results = torch.cat(results, dim=-1)
        outputs = torch.sum(torch.sum(results, dim=2), dim=1)

        align_feat1 = torch.cat(align_feat1, dim=2)
        align_feat2 = torch.cat(align_feat2, dim=2)
        align_features = torch.cat([align_feat1, align_feat2], dim=1)  # (B, 2, d_model * num_step)

        return outputs, logits, results.squeeze(), align_features

    def infer(self, x_sparse, x_seq, dense_feature, dense_coef, x_token):
        x = self.feature_embedding(x_sparse, x_seq, dense_feature, dense_coef)
        x = self.encoder(x)
        x = x.reshape(x.shape[0], self.num_layers, -1).permute(1, 0, 2)
        hidden_states = torch.split(x, [self.d_model, self.d_model], dim=2)
        hidden_states = (hidden_states[0].contiguous(), hidden_states[1].contiguous())

        emb = self.token_embedding(x_token)

        logits = None

        for t in range(self.num_step):
            output, (h_t, c_t) = self.lstm(emb, hidden_states)
            hidden_states = (h_t, c_t)
            logit = self.predictor(output)
            result = self.regressor(logit)

            if t == 0:
                logits = [logit]
                results = [result]
            else:
                logits.append(logit)
                results.append(result)
            emb = self.token_embedding.mix_forward(F.softmax(logit, dim=-1))

        # add regression for RQ error
        err = self.error_regressor(emb)
        results.append(err)

        logits = torch.cat(logits, dim=1)
        results = torch.cat(results, dim=-1)
        outputs = torch.sum(torch.sum(results, dim=2), dim=1)

        return outputs, logits, results
