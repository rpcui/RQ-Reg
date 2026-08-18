import os
import sys
import yaml
import json
import pickle
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.cikm16_dataset import CIKM16Dataset
from dataset.batch_func import *
from model.model import RQGenModel
from utils.criterion import *

import logging

def logging_config(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s  %(levelname)s: %(message)s'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("ERROR", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

def save_ckpt(model, optimizer, epoch, save_root):
    ckpt_path = os.path.join(save_root, 'model.pt')
    logging.info(f'Ckpt saved to {ckpt_path}')
    ckpt = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(ckpt, ckpt_path)

def save_pred_result(metrics, epoch, save_root):
    ckpt_path = os.path.join(save_root, f'epoch{epoch}_pred.npy')
    np.savez(ckpt_path, results=metrics["results"], alter=metrics["results_alter"])

def evaluate(model, val_data_loader, device, config):
    model.eval()

    all_pred, all_gt, losses = [], [], []
    target_pred, target_gt = [], []
    all_results, all_quant_code, all_results_alter = [], [], []
    with torch.no_grad():
        for _, inputs in enumerate(val_data_loader):
            x_sparse = inputs[0].to(device)
            x_seq = inputs[1].to(device)
            dense_feature = inputs[2].to(device)
            dense_coef = inputs[3].to(device)
            rq_code = inputs[4].to(device)
            rq_label = inputs[5].to(device)
            quant_code = inputs[6].to(device).squeeze().cpu().numpy()
            label = np.reshape(inputs[7], [-1])

            predictions, logits, results = model.infer(x_sparse, x_seq, dense_feature, dense_coef, rq_code)
            probs = torch.softmax(logits, dim=-1)
            results_alter_array = torch.sum(probs * model.vocabulary, dim=-1).cpu().numpy()

            loss = cross_entropy_loss(logits.permute(0, 2, 1), rq_label)
            losses.append(loss.item())

            pred_array = predictions.cpu().numpy()
            all_pred.append(pred_array)
            all_gt.append(label)

            results_array = results.cpu().numpy()
            all_results.append(results_array)
            all_quant_code.append(quant_code)
            all_results_alter.append(results_alter_array)

    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    mae = np.mean(np.abs(all_pred - all_gt))
    xauc = auc_score(all_pred, all_gt)
    metrics = {"loss": np.mean(losses), "MAE": mae, "XAUC": xauc}

    all_results = np.concatenate(all_results, axis=0).squeeze()  # (N, len)
    all_quant_code = np.concatenate(all_quant_code, axis=0)  # (N, len)
    results_cumsum = np.cumsum(all_results, axis=1)
    quant_code_cumsum = np.cumsum(all_quant_code, axis=1)
    all_results_alter = np.concatenate(all_results_alter, axis=0).squeeze()
    results_alter_cumsum = np.cumsum(all_results_alter, axis=1)

    seq_len = results_cumsum.shape[1]
    cumsum_mae_list = []
    cumsum_xauc_list = []
    for i in range(seq_len):
        pos_mae = np.mean(np.abs(results_cumsum[:, i] - all_gt))
        pos_xauc = auc_score(results_cumsum[:, i], all_gt)
        cumsum_mae_list.append(pos_mae)
        cumsum_xauc_list.append(pos_xauc)

    metrics["cumsum_MAE_list"] = cumsum_mae_list
    metrics["cumsum_XAUC_list"] = cumsum_xauc_list

    results_last = all_results[:, -1]  # shape=(N,)
    quant_code_last = all_quant_code[:, -1]  # shape=(N,)
    mae_err = np.mean(np.abs(results_last - quant_code_last))
    xauc_err = auc_score(results_last, quant_code_last)
    metrics["MAE_err"] = mae_err
    metrics["XAUC_err"] = xauc_err

    metrics["predictions"] = np.reshape(all_pred, [-1, 1])
    metrics["results"] = results_cumsum
    metrics["results_alter"] = results_alter_cumsum

    return metrics

def train_and_eval(model, optimizer, train_data_loader, val_data_loader, device, config):
    model.train()

    losses, errors = [], []
    target_errors = []
    for epoch in range(config['num_epoch']):
        sampling_ratio = 1.0 / (1.0 + np.exp(config['tau'] * (config['pivot'] - epoch / config['num_epoch'])))
        best_metric = np.inf
        for batch_idx, inputs in enumerate(train_data_loader):
            x_sparse = inputs[0].to(device)
            x_seq = inputs[1].to(device)
            dense_feature = inputs[2].to(device)
            dense_coef = inputs[3].to(device)
            rq_code = inputs[4].to(device)
            rq_label = inputs[5].to(device)
            quant_code = inputs[6].to(device).squeeze()
            label = inputs[7].to(device).squeeze()

            predictions, logits, results, align_features = model(x_sparse, x_seq, dense_feature, dense_coef, rq_code, sampling_ratio)
            ce_loss = cross_entropy_loss(logits.permute(0, 2, 1), rq_label)
            results_cumsum = torch.stack((
                results[:, 0],
                results[:, 0].detach() + results[:, 1],
                torch.sum(results[:, 0: 2], axis=1).detach() + results[:, 2],
                torch.sum(results[:, 0: 3], axis=1).detach() + results[:, 3]), dim=1)
            codes_cumsum = torch.cumsum(quant_code, dim=1)
            huber_cumsum = huber_loss(results_cumsum, codes_cumsum, config['delta'])
            rnc_loss = RnC_loss(align_features[:, 0], label.unsqueeze(-1), config['temperature'])

            mae = l1_loss(predictions, label)
            loss = ce_loss + config['lambda1'] * huber_cumsum + config['lambda2'] * rnc_loss

            losses.append(ce_loss.item())
            errors.append(mae.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % config['log_interval'] == 0:
                logging.info('Train epoch: {} [{} / {} ({:.0f}%)], Loss: {:.5f}, MAE: {:.4f}'.format(
                    epoch, (batch_idx + 1) * config['batch_size'], len(train_data_loader.dataset),
                    100. * (batch_idx + 1) / len(train_data_loader), np.mean(losses), np.mean(errors)))
                losses, errors = [], []

        metrics = evaluate(model, val_data_loader, device, config)
        model.train()
        logging.info('Train epoch: {}, eval loss: {:.5f}, MAE: {:.3f}, XAUC: {:.4f}'.format(
            epoch, metrics['loss'], metrics['MAE'], metrics['XAUC']))

        if metrics['MAE'] < best_metric:
            best_metric = metrics['MAE']
            save_ckpt(model, optimizer, epoch, config['save_root'])
            if config['debug']:
                save_pred_result(metrics, epoch, config['save_root'])

def main():
    parser = argparse.ArgumentParser(description='Continuous value prediction')
    parser.add_argument('--dataset', type=str, required=True, choices=('cikm16', 'kuairec'), help='Dataset: [cikm16 / kuairec]')
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    args = parser.parse_args()

    config_file = f'config_{args.dataset}.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['train']['save_root'] = os.path.join(config['train']['save'], args.dataset, args.exp)
    if not os.path.exists(config['train']['save_root']):
        os.makedirs(config['train']['save_root'])

    log_file = os.path.join('logs/', f'{args.dataset}_{args.exp}.log')
    logging_config(log_file)

    script_path = os.path.abspath(__file__)
    logging.info(f"\n============ Script: {script_path} ============\n")
    with open(script_path, 'r', encoding='utf-8') as f:
        logging.info(f.read())
    logging.info("\n============ End of Script ============\n")

    logging.info("\n============\n")
    logging.info("Arguments: ")
    logging.info(json.dumps(args.__dict__, indent=4))
    logging.info("\n============\n")
    logging.info("Config: ")
    logging.info(json.dumps(config, indent=4))
    logging.info("\n============\n")

    cfg = config['data']
    rq_model_path = os.path.join(
        cfg['code_path'],
        "-".join([f'{cfg["method"]}{codebook_size}' for codebook_size in cfg["codebook_size"]]),
        'model.pkl')
    with open(rq_model_path, 'rb') as f:
        config['data']['codebooks'] = pickle.load(f)
        vocabulary = [0.0, 0.0]
        for codebook in config['data']['codebooks']:
            vocabulary += list(np.squeeze(codebook))
        config['data']['vocabulary'] = vocabulary

    config['model']['input_dim'] = sum([len(v) for v in cfg['embedding_size'].values()]) * config['model']['embedding_dim']

    if args.dataset == 'cikm16':
        train_data = CIKM16Dataset(config['data'], phase='train')
        val_data = CIKM16Dataset(config['data'], phase='val')

    train_data_loader = DataLoader(train_data, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4, collate_fn=train_collate_fn)
    val_data_loader = DataLoader(val_data, batch_size=config['train']['batch_size'], shuffle=False, num_workers=4, collate_fn=val_collate_fn)

    device = torch.device(config['train']['device'])
    model = RQGenModel(config['data'], config['model']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])

    train_and_eval(model, optimizer, train_data_loader, val_data_loader, device, config['train'])


if __name__ == "__main__":
    main()
