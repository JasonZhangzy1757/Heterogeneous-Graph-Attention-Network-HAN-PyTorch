import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model_RGCN import RGCN
from utils import load_data_RGCN, accuracy, knn_classifier


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
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='The dataset to use for the model.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load the data
    pyg_data = load_data_RGCN(args.dataset)

    model = RGCN(num_nodes=pyg_data.x.size(1),
                 num_relations=pyg_data.edge_type.max().item() + 1,
                 hidden_dim=args.hidden_dim,
                 num_classes=pyg_data.y.unique().size(0))

    if args.cuda:
        model.cuda()
        pyg_data.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    best_loss_val = float('inf')
    patience = args.patience
    counter = 0
    idx_train = pyg_data.train_mask
    idx_val = pyg_data.val_mask
    idx_test = pyg_data.test_mask
    labels = pyg_data.y

    for epoch in range(args.epochs):
        model.train()
        output = model(pyg_data)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(pyg_data)
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
    output = model(pyg_data)
    X = output[idx_test].detach().cpu().numpy()
    y = labels[idx_test].detach().cpu().numpy()
    knn_classifier(X, y, seed=args.seed)

