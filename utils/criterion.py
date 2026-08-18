import math
import numpy as np
import torch
import torch.nn.functional as F

def cross_entropy_loss(logit, y):
    loss = F.cross_entropy(logit, y.long(), reduction='none')
    d_loss = torch.mean(loss)
    return d_loss

def l1_loss(y_hat, y):
    return F.l1_loss(y_hat, y, reduction='mean')

def mse_loss(y_hat, y):
    return F.mse_loss(y_hat, y, reduction='mean')

def huber_loss(y_hat, y, delta=10.0):
    return F.huber_loss(y_hat, y, delta=delta, reduction='mean')

def RnC_loss(features, labels, temperature=2.0):
    # features: [bs, feat_dim]
    # labels: [bs, label_dim]
    label_diffs = torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
    logits = -(features[:, None, :] - features[None, :, :]).norm(2, dim=-1).div(temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits -= logits_max.detach()
    exp_logits = logits.exp()

    n = logits.shape[0]  # n = bs

    # remove diagonal
    logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
    exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
    label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

    loss = 0.0
    for k in range(n - 1):
        pos_logits = logits[:, k]  # bs
        pos_label_diffs = label_diffs[:, k]  # bs
        neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [bs, bs - 1]
        pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # bs
        loss += - (pos_log_probs / (n * (n - 1))).sum()
    return loss

def inverse_pairs(data):
    if not data:
        return 0
    if len(data) == 1:
        return 0
    def merge(tuple_fir, tuple_sec):
        array_before = tuple_fir[0]
        cnt_before = tuple_fir[1]
        array_after = tuple_sec[0]
        cnt_after = tuple_sec[1]
        cnt = cnt_before + cnt_after
        flag = len(array_after) - 1
        array_merge = []
        for i in range(len(array_before) - 1, -1, -1):
            while array_before[i] <= array_after[flag] and flag >= 0:
                array_merge.append(array_after[flag])
                flag -= 1
            if flag == -1:
                break
            else:
                array_merge.append(array_before[i])
                cnt += (flag + 1)
        if flag == -1:
            for j in range(i, -1, -1):
                array_merge.append(array_before[j])
        else:
            for j in range(flag, -1, -1):
                array_merge.append(array_after[j])
        return array_merge[:: -1], cnt

    def mergesort(array):
        if len(array) == 1:
            return (array, 0)
        cut = math.floor(len(array) / 2)
        tuple_fir=mergesort(array[: cut])
        tuple_sec=mergesort(array[cut: ])
        return merge(tuple_fir, tuple_sec)

    return mergesort(data)[1]

def auc_score(labels, pres):
    label_preds = zip(labels.reshape(-1), pres.reshape(-1))
    sorted_label_preds = sorted(
        label_preds, key=lambda lc: lc[1], reverse=True)
    label_preds_len = len(sorted_label_preds)
    pairs_cnt = label_preds_len * (label_preds_len-1) / 2

    labels_sort = [ele[0] for ele in sorted_label_preds]
    total_positive = inverse_pairs(labels_sort)
    xauc = total_positive / pairs_cnt
    return xauc
