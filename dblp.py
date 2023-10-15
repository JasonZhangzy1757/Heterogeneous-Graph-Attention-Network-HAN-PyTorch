import os
import dgl
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, generate_mask_tensor, idx2mask


class DBLP4057Dataset(DGLDataset):
    """
        The DBLP dataset processed by the Heterogeneous Graph Attention Network
        The code was retrieved from https://github.com/ZZy979/pytorch-tutorial/blob/master/gnn/data/dblp.py
    """
    def __init__(self):
        super().__init__('DBLP4057', 'https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg')

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), self.gs)

    def load(self):
        self.gs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        for g in self.gs:
            for k in ('train_mask', 'val_mask', 'test_mask'):
                g.ndata[k] = g.ndata[k].bool()

    def process(self):
        data = sio.loadmat('./data/DBLP4057_GAT_with_idx.mat')
        apa_g = dgl.graph(data['net_APA'].nonzero())
        apcpa_g = dgl.graph(data['net_APCPA'].nonzero())
        aptpa_g = dgl.graph(data['net_APTPA'].nonzero())
        self.gs = [apa_g, apcpa_g, aptpa_g]

        features = torch.from_numpy(data['features']).float()
        labels = torch.from_numpy(data['label'].nonzero()[1])
        num_nodes = data['label'].shape[0]
        train_mask = generate_mask_tensor(idx2mask(data['train_idx'][0], num_nodes))
        val_mask = generate_mask_tensor(idx2mask(data['val_idx'][0], num_nodes))
        test_mask = generate_mask_tensor(idx2mask(data['test_idx'][0], num_nodes))
        for g in self.gs:
            g.ndata['feat'] = features
            g.ndata['label'] = labels
            g.ndata['train_mask'] = train_mask
            g.ndata['val_mask'] = val_mask
            g.ndata['test_mask'] = test_mask

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
        return 4