import numpy as np

import torch
import torch.nn as nn
import random
import dgl
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.conv import GINConv
import torch.nn as F

import os
from dgl.data import DGLDataset
from dgl import load_graphs
from dgl.data.utils import  load_info
import math

import argparse
import logging

from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="GNN for network classification", allow_abbrev=False)
    parser.add_argument("--dataset_path", type=str, default="../data_folder/data", help="Path to dataset")
    parser.add_argument("--test_path", type=str, default="../data_folder/test")
    parser.add_argument("--weight_path", type=str, default="../weights", help="Output path")
    parser.add_argument("--device", type=str, default="cpu", help="Device cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay of the learning rate over epochs for the optimizer")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden size, number of neuron in every hidden layer but could change for currten type of networks")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout ratio")
    parser.add_argument("--epochs", type=int, default=100, help="Max number of training epochs")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of conv layers")
    parser.add_argument("--print_every", type=int, default=10, help="Print train log every k epochs, -1 for silent training")
    parser.add_argument("--output_activation", type=str, default="Identity", help="Output activation function")
    parser.add_argument("--optimizer_name", type=str, default="Adam", help="Optimizer type default adam")
    parser.add_argument("--loss_name", type=str, default='MSELoss', help="Choose loss function correlated to the optimization function")

    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, use CPU for training.")
        args.device = "cpu"

    return args

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms(True)

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
        self.relu = nn.ReLU()
    def forward(self, x):
        h = x
        h = self.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)
    
class GIN(nn.Module):
    def __init__(self, in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers = 5,
                 dropout=0.,
                 output_activation = 'log_softmax'):

        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.output_activation = output_activation

        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        mlp = MLP(hidden_dim, hidden_dim, 1)
        self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
        self.batch_norms.append(nn.BatchNorm1d(1))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers+1):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(in_dim, out_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, out_dim))
        self.linear_prediction.append(nn.Linear(1, out_dim))
        self.drop = nn.Dropout(dropout)
        #self.mlp = MLP(hidden_dim, hidden_dim, out_dim)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets
        self.relu = nn.ReLU()
        self.output_activation = getattr(nn, self.output_activation)(dim=-1)

    def forward(self, g, args):
        # list of hidden representation at each layer (including the input layer)
        h = g.ndata["feat"]
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = self.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        pooled_h_list = []
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            pooled_h_list.append(pooled_h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

        #score_over_layer = self.mlp(score_over_layer)
        return  self.output_activation(score_over_layer)
    

def train(model: torch.nn.Module, optimizer, trainloader, args):
    model.train()
    total_loss = 0.0
    num_graphs = 0
    
    loss_func = getattr(F, args.loss_name)(reduction="sum")
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        num_graphs += args.batch_size
    
        out = model(batch_graphs, args)
        loss = loss_func(out, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / num_graphs

@torch.no_grad()
def test_regression(model: torch.nn.Module, loader, args):
    model.eval()
    loss = 0.0
    num_graphs = 0
    loss_func = getattr(F, args.loss_name)(reduction="sum")
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += args.batch_size
        out = model(batch_graphs, args)
        loss += loss_func(out, batch_labels).item()

    return loss / num_graphs

class GraphDataset(DGLDataset):

    def __init__(self, graphs=None, labels=None, device='cpu'):
        self.graphs = graphs
        self.labels = labels 
        self.device = device
        self.data_path = None
        if labels != None:
          self.dim_nfeats = len(self.graphs[0].ndata)
          self.gclasses = len(self.labels.unique())
          if self.device == 'cuda':
            self.graphs = [g.to(self.device) for g in self.graphs]
            self.labels = self.labels.to(self.device)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def statistics(self):
        return self.dim_nfeats, self.gclasses, self.device

    def load(self, data_path, args):
        '''
        Loads the processed data from disk as .bin and .pkl files. The processed data consists of the graph data and the corresponding labels.
        '''
        # Load the graph data and labels from the .bin file
        graph_path = os.path.join('{}/dgl_graph.bin'.format(data_path))
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        # Load the other information about the dataset from the .pkl file
        info_path = os.path.join('{}/info.pkl'.format(data_path))
        self.gclasses = load_info(info_path)['gclasses']
        self.dim_nfeats = load_info(info_path)['dim_nfeats']
        #self.device = load_info(info_path)['device']
        self.data_path = data_path
        self.labels = torch.load('../data_folder/data/properties_labels.pt')
        self.labels = self.labels[4]
        if self.device == 'cuda':
            self.graphs = [g.to(self.device) for g in self.graphs]
            self.labels = self.labels.to(self.device)        

    def has_cache(self):
        '''
        Checks if the processed data has been saved to disk as .bin and .pkl files.
        '''
        # Check if the .bin and .pkl files for the processed data exist in the directory
        graph_path = os.path.join(f'{self.data_path}/dgl_graph.bin')
        info_path = os.path.join(f'{self.data_path}/info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    def add_self_loop(self):
        for graph in self.graphs:
            graph = graph.add_self_loop()
    
    def add_ones_feat(self):
        for g in self.graphs:
            g.ndata['feat'] = torch.ones(g.num_nodes(), 1).float().to(self.device)

def main(args, seed):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    set_random_seed(seed)
    dataset = GraphDataset(device=args.device)
    dataset.load(args.dataset_path, args)
    dataset.add_ones_feat()
    
    num_training = int(len(dataset) * 0.9)
    num_val = int(len(dataset) * 0.)
    num_test = len(dataset) - num_val - num_training
    generator = torch.Generator().manual_seed(seed)
    train_set, _, test_set = random_split(dataset, [num_training, num_val, num_test], generator=generator)

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    # Step 2: Create model =================================================================== #
    num_feature, num_classes, _ = dataset.statistics()
    args.num_feature = int(num_feature)
    args.num_classes = int(num_classes)
    #set_random_seed(seed)
    weight_path = f"{args.weight_path}/trial_{seed+1}_{args.hidden_dim}_{args.num_layers}_{args.lr}_{args.weight_decay}_{args.dropout}_{args.output_activation}_weights.pth"

    model = GIN(
        in_dim=1,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_activation = args.output_activation
    ).to(args.device)

    # Try to load model weights
    try:
        model.load_state_dict(torch.load(weight_path))
        print(f"Weights loaded successfully.")
    except FileNotFoundError:
        print(f"Could not find weights, initializing model with random weights, and saving it.")
        torch.save(model.state_dict(), weight_path)

    # Step 3: Create training components ===================================================== #
    if hasattr(torch.optim, args.optimizer_name):
        optimizer = getattr(torch.optim, args.optimizer_name)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # Replace `parameters` with your specific parameters
    else:
        print(f"Optimizer '{args.optimizer_name}' not found in torch.optim.")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Step 4: training epoches =============================================================== #
    for e in range(args.epochs):
        train_loss = train(model, optimizer, train_loader, args)
        scheduler.step()
        

        if (e + 1) % args.print_every == 0:
            log_format = ("Epoch {}: loss={:.4f}")
            print(log_format.format(e + 1, train_loss))
    test_loss1 = test_regression(model, test_loader, args)
    print(f'test1 loss : {test_loss1}')
args = parse_args()

main(args, 1)
main(args, 1)
