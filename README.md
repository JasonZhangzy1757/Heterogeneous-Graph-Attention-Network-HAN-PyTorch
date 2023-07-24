# Heterogeneous-Graph-Attention-Network-HAN-PyTorch

This is a proof of concept simple version of HAN implementation using pytorch framework. The authors' original code can be found [here](https://github.com/Jhy1993/HAN).

If you find this work helpful for your research, you could cite the original paper as the following:
```
@inproceedings{wang2019heterogeneous,
  title={Heterogeneous graph attention network},
  author={Wang, Xiao and Ji, Houye and Shi, Chuan and Wang, Bai and Ye, Yanfang and Cui, Peng and Yu, Philip S},
  booktitle={The world wide web conference},
  pages={2022--2032},
  year={2019}
}
```

This implementation is also inspired by the [dgl implemenation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/han) and the earlier pytorch implementation of GAT by [Diego999](https://github.com/Diego999/pyGAT).

## Usage

1) Download the DBLP preprocessed data from the author's repo and put the data under a new data directory.
2) `python main.py` for reproducing HAN's work.


## Performance

The performance on the DBLP classification task compared with the original paper.

|                     |    Paper          | Pytorch        |
| ------------------- | --------------    | -------------- |
| Macro-F1 80%        |    93.08          | 93.28          |
| Macro-F1 60%        |    92.80          | 92.07          |
| Macro-F1 40%        |    92.40          | 92.27          |
| Macro-F1 20%        |    92.24          | 92.00          |
| Micro-F1 80%        |    93.99          | 94.06          |
| Micro-F1 60%        |    93.70          | 93.09          |
| Micro-F1 40%        |    93.30          | 93.12          |
| Micro-F1 20%        |    93.11          | 92.91          |
