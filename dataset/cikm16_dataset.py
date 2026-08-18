import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def l1_distance(x, y):
    return np.abs(x - y)

class CIKM16Dataset(Dataset):
    def __init__(self, config, phase='train'):
        self.config = config

        if phase == 'train':
            filename = os.path.join(config['path'], 'train_data_new.csv')
        elif phase == 'val':
            filename = os.path.join(config['path'], 'val_data_new.csv')

        assert os.path.exists(filename)
        df_data = pd.read_csv(filename, usecols=['session_id', 'duration', 'query_length', 'item_length'] +
            [f'query{i}' for i in range(10)] + [f'item{i}' for i in range(10)])

        self.data = df_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data.iloc[index]

        sparse_columns = self.config['feature_col']['sparse']
        seq_columns = self.config['feature_col']['seq']
        label_columns = self.config['feature_col']['label']

        sparse_feature = data[sparse_columns].values.astype(int)
        seq_feature = data[seq_columns].values.astype(int)

        label = data[label_columns].values.astype(float)
        rq_code, rq_quant_code = self.get_rq_code(label)

        return sparse_feature, seq_feature, np.array([]), np.array([]), rq_code, rq_quant_code, label

    def get_rq_code(self, value):
        codebooks = self.config['codebooks']
        codes = [1]
        quant_codes = [0.0]
        for i, codebook in enumerate(codebooks):
            offset = sum([len(codebook) for codebook in codebooks[: i]]) + 2
            if np.abs(value) < 1e-4:
                codes.append(0)
                quant_codes.append(0.0)
            else:
                codebook = np.squeeze(codebook)
                err = l1_distance(value, codebook)
                code = np.argmin(err)
                codes.append(code + offset)
                quant_codes.append(codebook[code])
                value = value - codebook[code]
        quant_codes.append(float(value))  # add quantization error
        return np.array(codes).astype(int), np.array(quant_codes).astype(float)
