# RQ-Reg
Code release for \[CIKM'26\] RQ-Reg: A Residual-Quantization-Based Framework for Continuous Value Prediction in Recommender Systems

### Training

For CIKM16 dataset processing, please refer to `https://github.com/jackielinxiao/TPM/blob/main/data_process.py`.

To get residual quantization (RQ) K-means codebook for training, run the script `rq_kmeans/get_rq_codebook.py`.

To train the RQ-Reg model, please update the config file with local paths, and then run the following code:

```
python main.py --dataset cikm16 --exp <exp_name>
```

### Reference

If you find this repository useful, please cite the following paper:

```
@inproceedings{cui2026rqreg,
  author    = {Cui, Runpeng and Sun, Zhipeng and Lu, Chi and Jiang, Peng},
  title     = {{RQ-Reg:} A Residual-Quantization-Based Framework for Continuous Value
               Prediction in Recommender Systems},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026}
}
```
