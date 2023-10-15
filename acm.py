import os
import torch
import pickle
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import _get_dgl_url, download, save_graphs, load_graphs, \
    generate_mask_tensor, idx2mask


class ACM3025Dataset(DGLDataset):
    """
    The ACM dataset processed by the Heterogeneous Graph Attention Network
    The code was retrieved from https://github.com/ZZy979/pytorch-tutorial/blob/master/gnn/data/acm.py
    """

    def __init__(self):
        super().__init__('ACM3025', _get_dgl_url('dataset/ACM3025.pkl'))

    def download(self):
        file_path = os.path.join(self.raw_dir, 'ACM3025.pkl')
        if not os.path.exists(file_path):
            download(self.url, path=file_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), self.gs)

    def load(self):
        self.gs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        for g in self.gs:
            for k in ('train_mask', 'val_mask', 'test_mask'):
                g.ndata[k] = g.ndata[k].bool()

    def process(self):
        with open(os.path.join(self.raw_dir, 'ACM3025.pkl'), 'rb') as f:
            data = pickle.load(f)
        features = torch.from_numpy(data['feature'].todense()).float()  # (3025, 1870)
        labels = torch.from_numpy(data['label'].todense()).long().nonzero(as_tuple=True)[1]  # (3025)

        # Adjacency matrices for meta-path based neighbors
        # (Mufei): I verified both of them are binary adjacency matrices with self loops
        author_g = dgl.from_scipy(data['PAP'])
        subject_g = dgl.from_scipy(data['PLP'])
        self.gs = [author_g, subject_g]

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

