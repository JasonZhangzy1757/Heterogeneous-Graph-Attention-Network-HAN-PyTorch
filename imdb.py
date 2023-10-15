
import os

import dgl
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import download, save_graphs, load_graphs, generate_mask_tensor, idx2mask


class IMDb5kDataset(DGLDataset):
    """
    The IMDB dataset processed by the Heterogeneous Graph Attention Network
    The code was retrieved from https://github.com/ZZy979/pytorch-tutorial/blob/master/gnn/data/imdb.py
    """

    def __init__(self):
        super().__init__('imdb5k', 'https://pan.baidu.com/s/199LoAr5WmL3wgx66j-qwaw')

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), self.gs)

    def load(self):
        self.gs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        for g in self.gs:
            for k in ('train_mask', 'val_mask', 'test_mask'):
                g.ndata[k] = g.ndata[k].bool()

    def process(self):
        data = sio.loadmat('./data/imdb5k.mat')
        mam_g = dgl.graph(data['MAM'].nonzero())
        mdm_g = dgl.graph(data['MDM'].nonzero())
        # mym_g = dgl.graph(data['MYM'].nonzero())
        self.gs = [mam_g, mdm_g]

        features = torch.from_numpy(data['feature']).float()
        num_nodes = features.shape[0]
        labels = torch.full((num_nodes,), -1, dtype=torch.long)
        idx, label = data['label'].nonzero()
        labels[idx] = torch.from_numpy(label)
        train_mask = generate_mask_tensor(idx2mask(data['train_idx'][0], num_nodes))
        val_mask = generate_mask_tensor(idx2mask(data['val_idx'][0], num_nodes))
        test_mask = generate_mask_tensor(idx2mask(data['test_idx'][0], num_nodes))
        for g in self.gs:
            g.ndata['feat'] = features
            g.ndata['label'] = labels
            g.ndata['train_mask'] = train_mask
            g.ndata['val_mask'] = val_mask
            g.ndata['test_mask'] = test_mask
        for i, g in enumerate(self.gs):
            g.edata['type'] = torch.full((g.number_of_edges(),), i, dtype=torch.long)

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.gs

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 3