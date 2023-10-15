# Heterogeneous-Graph-Attention-Network-HAN-PyTorch

This is a proof of concept of HAN implementation using pytorch framework. The authors' original code can be found [here](https://github.com/Jhy1993/HAN).

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

This implementation is also inspired by the [dgl implemenation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/han) and the earlier pytorch implementation of GAT by [Diego999](https://github.com/Diego999/pyGAT). The data processing is copied from [ZZy979](https://github.com/ZZy979). Please also check his other brilliant works in his page. 

## Usage

1) Download the data [here](https://drive.google.com/drive/folders/13RcthEaCjg2yILIWZgzk-Xs4si-mlQZd?usp=sharing) and put the data under a new data directory. If you don't have access to Google Drive, you could also check [ZZy979's](https://github.com/ZZy979) work. 
2) `python main.py` for reproducing HAN's work.
3) `python RGCN_baseline.py` for adding on an RGCN baseline.
4) Use `--dataset` to specify the dataset you hope to run against. Currently it's `DBLP` by default. The options could be: `ACM` or `IMDB`. 


## Performance
![Results](https://github.com/JasonZhangzy1757/Heterogeneous-Graph-Attention-Network-HAN-PyTorch/assets/56742253/f0c9db4f-d9da-44ce-8b0b-b9e5e3f4e17b)


