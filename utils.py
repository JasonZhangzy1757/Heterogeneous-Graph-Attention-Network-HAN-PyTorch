import scipy.io as sio
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def load_data(path='./data/DBLP4057_GAT_with_idx.mat'):
    # Load the preprocessed DBLP the author provided.
    data = sio.loadmat(path)

    features, labels = data['features'], data['label']
    # Three adjacency matrix for three meta paths the dataset contains.
    net_APTPA, net_APCPA, net_APA = data['net_APTPA'], data['net_APCPA'], data['net_APA']
    idx_train, idx_val, idx_test = data['train_idx'], data['val_idx'], data['test_idx']

    net_APTPA = torch.LongTensor(net_APTPA)
    net_APCPA = torch.LongTensor(net_APCPA)
    net_APA = torch.LongTensor(net_APA)
    meta_path_list = [net_APTPA, net_APCPA, net_APA]

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train).squeeze()
    idx_val = torch.LongTensor(idx_val).squeeze()
    idx_test = torch.LongTensor(idx_test).squeeze()

    return features, meta_path_list, labels, idx_train, idx_val, idx_test


def accuracy(preds, labels):
    _, indices = torch.max(preds, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    return (prediction == labels).sum() / len(prediction)


def knn_classifier(X, y, seed=42, k=5, time=10):
    """
    Adapted from Jhy1993/HAN
    A typical KNN classifier
    """
    for split in [0.2, 0.4, 0.6, 0.8]:
        micro_list = []
        macro_list = []
        for i in range(time):
            # Shuffle the test set to make sure each time difference sets of data are selected.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split, random_state=seed)

            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_micro = f1_score(y_test, y_pred, average='micro')
            macro_list.append(f1_macro)
            micro_list.append(f1_micro)

        print('KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
            time, split, k, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list)))
