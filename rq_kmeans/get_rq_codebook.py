import os
import time
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from res_cluster import ResCluster
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(format='%(asctime)s  %(levelname)s: %(message)s',
                    level=logging.INFO, stream=sys.stderr)

config = {
    "data_path": "",  # path for dataset
    "max_workers": 1,
    "column": "duration",
    "max_clip_value": 6.0,
    "model": {
        "method": "kmeans",
        "n_layers": 3,
        "codebook_size": [48, 48, 48],
        "dim": 1,
        "lambda": 2e-6,
    },
}

def load_one_file(fn, expected_dim, column, max_clip_value):
    df = pd.read_csv(fn)

    if column not in df.columns:
        raise ValueError(f"Column {column} is not in {fn}")

    emb_col = df[column]
    n = len(emb_col)
    if n == 0:
        return np.empty((0, expected_dim))
    out = np.reshape(emb_col.values.astype(float), (n, expected_dim))
    out[out >= max_clip_value] = max_clip_value
    return out

def load_continuous_value_from_files(data_path,
                                     max_workers=1,
                                     expected_dim=1,
                                     column="duration",
                                     max_clip_value=1e3):
    files = [os.path.join(data_path, "train_data.csv")]
    tensors = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                load_one_file,
                fn, expected_dim, column, max_clip_value
            )
            for fn in files
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            t = fut.result()
            if t.size > 0:
                tensors.append(t)

    return np.concatenate(tensors, axis=0)  # [N_total, expected_dim]

def main():
    config["save_path"] = os.path.join(config["data_path"], "models")

    all_values = load_continuous_value_from_files(config["data_path"],
        config["max_workers"],
        config["sample_seed"],
        config["model"]["dim"],
        config["column"],
        config["max_clip_value"])

    model_configs = [
        {"n_layers": 3, "codebook_size": [48, 48, 48], "lambda": 2e-6, "method": "kmeans"},
    ]

    for model_cfg in model_configs:
        current_cfg = {
            "method": model_cfg["method"],
            "dim": config["model"]["dim"],

            "n_layers": model_cfg["n_layers"],
            "codebook_size": model_cfg["codebook_size"],
            "lambd": model_cfg["lambda"]
        }

        logging.info(f"Training: method={current_cfg['method']}, n_layers={current_cfg['n_layers']}, codebook_size={current_cfg['codebook_size']}, lambda={current_cfg['lambd']}")

        model = ResCluster(**current_cfg)
        model.train(all_values)

        codes, quant_codes, num_samples = model.encode(all_values)
        for n in range(current_cfg["n_layers"]):
            logging.info(f"Codebook Layer: {n + 1}")
            for k in range((current_cfg["codebook_size"][n] - 1) // 8 + 1):
                code_strs = ["{:16.4f}".format(float(val)) for val in model.codebooks[n][k * 8: (k + 1) * 8]]
                code_string = "Code : " + "".join(code_strs)
                logging.info(code_string)

            for k in range((current_cfg["codebook_size"][n] - 1) // 8 + 1):
                code_strs = ["{:16.4f}".format(float(val) / float(num_samples[n, 0])) for val in list(num_samples[n, k * 8 + 1: (k + 1) * 8 + 1])]
                code_string = "Ratio: " + "".join(code_strs)
                logging.info(code_string)

        quantized_values = np.sum(quant_codes, axis=1)
        error = quantized_values - all_values[:, 0]
        logging.info(f"max error: {np.max(error)}, min error: {np.min(error)}, MAE: {np.mean(np.abs(error))}, MAPE: {np.mean(np.abs(error / (all_values[:, 0] + 1e-10)))}")

        method = current_cfg["method"]
        if type(method) == str:
            method = [method] * current_cfg["n_layers"]
        save_string = '-'.join([f"{u}{v}" for u, v in zip(method, current_cfg["codebook_size"])])

        save_path = os.path.join(config["save_path"], save_string)
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model.codebooks, f)
        logging.info(f"Codebook saved to: {model_path}")

if __name__ == "__main__":
    main()