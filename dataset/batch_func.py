import torch
import numpy as np

# padding
def seq_padding(seq, target_size):
    seq_length = seq.shape[0]
    padding_size = target_size - seq_length
    if len(seq.shape) == 2:
        padding_seq = np.pad(np.array(seq), ((0, padding_size), (0, 0)), mode = 'constant', constant_values = 0)
    elif len(seq.shape) == 1:
        padding_seq = np.pad(np.array(seq), ((0, padding_size)), mode = 'constant', constant_values = 0)
    else:
        raise ValueError('input seq shape length must be 1 or 2. get : {}'.format(len(seq.shape)))
    return padding_seq

def train_collate_fn(batch):
    sparse_feature = torch.tensor(
        np.array([item[0] for item in batch]), dtype=torch.int)
    seq_feature = torch.tensor(
        np.array([item[1] for item in batch]), dtype=torch.int)
    dense_feature = torch.tensor(
        np.array([item[2] for item in batch]), dtype=torch.int)
    dense_coef = torch.tensor(
        np.array([item[3] for item in batch]), dtype=torch.float)

    rq_code = torch.tensor(
        np.array([item[4][: -1] for item in batch]), dtype=torch.int)
    rq_label = torch.tensor(
        np.array([item[4][1: ] for item in batch]), dtype=torch.int)
    quant_code = torch.tensor(
        np.array([item[5][1: ] for item in batch]), dtype=torch.float)
    label = torch.tensor(
        np.array([item[6] for item in batch]), dtype=torch.float)
    return sparse_feature, seq_feature, dense_feature, dense_coef, rq_code, rq_label, quant_code, label

def val_collate_fn(batch):
    sparse_feature = torch.tensor(
        np.array([item[0] for item in batch]), dtype=torch.int)
    seq_feature = torch.tensor(
        np.array([item[1] for item in batch]), dtype=torch.int)
    dense_feature = torch.tensor(
        np.array([item[2] for item in batch]), dtype=torch.int)
    dense_coef = torch.tensor(
        np.array([item[3] for item in batch]), dtype=torch.float)

    rq_code = torch.tensor(
        np.array([item[4][[0]] for item in batch]), dtype=torch.int)
    rq_label = torch.tensor(
        np.array([item[4][1: ] for item in batch]), dtype=torch.int)
    quant_code = torch.tensor(
        np.array([item[5][1: ] for item in batch]), dtype=torch.float)
    label = np.array([item[6] for item in batch])
    return sparse_feature, seq_feature, dense_feature, dense_coef, rq_code, rq_label, quant_code, label
