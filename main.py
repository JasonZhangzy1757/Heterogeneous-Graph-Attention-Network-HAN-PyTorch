import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model import HAN
from utils import load_data, accuracy, knn_classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=88,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='Number of hidden dimension.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads for node-level attention.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha for the leaky_relu.')
    parser.add_argument('--q_vector', type=int, default=128,
                        help='The dimension for the semantic attention embedding.')
    parser.add_argument('--patience', type=int, default=100,
                        help='Number of epochs with no improvement to wait before stopping')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load the data
    # features: (N, F), meta_path_list: (M, N, N), labels: (N, 1)
    features, meta_path_list, labels, idx_train, idx_val, idx_test = load_data()
    model = HAN(feature_dim=features.shape[1],
                hidden_dim=args.hidden_dim,
                num_classes=int(labels.max()) + 1,
                dropout=args.dropout,
                num_heads=args.num_heads,
                alpha=args.alpha,
                q_vector=args.q_vector)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        meta_path_list = [meta_path.cuda() for meta_path in meta_path_list]
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    best_loss_val = float('inf')
    patience = args.patience
    counter = 0

    for epoch in range(args.epochs):
        model.train()
        output = model(features, meta_path_list)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(features, meta_path_list)
            loss_val = criterion(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val),
                  'acc_val: {:.4f}'.format(acc_val))

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                # Save the best model state if needed
                torch.save(model.state_dict(), 'best_model.pth')
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    # Load the best model state
    model.load_state_dict(torch.load('best_model.pth'))

    model.eval()
    output = model(features, meta_path_list)
    X = output[idx_test].detach().cpu().numpy()
    y = labels[idx_test].detach().cpu().numpy()
    knn_classifier(X, y, seed=args.seed)
