#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:42:16 2021

@author: Gong Dongsheng
"""

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False, device='cuda'):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        # support = torch.mm(self.dropout(inputs), self.weight)
        support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_layers, n_features, hidden_dim, dropout, n_classes,device='cuda'):
        super(GCN, self).__init__()
        if n_layers == 1:
            self.first_layer = GraphConvolution(n_features, n_classes, dropout, device=device)
        else:
            self.first_layer = GraphConvolution(n_features, hidden_dim, dropout, device=device)
            self.last_layer = GraphConvolution(hidden_dim, n_classes, dropout, device=device)
            if n_layers > 2:
                self.gc_layers = nn.ModuleList([
                    GraphConvolution(hidden_dim, hidden_dim, 0) for _ in range(n_layers - 2)
                ])

        self.lstm = nn.LSTM(1, hidden_dim, 2, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        self.device=device
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(32, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim//2, n_classes))
    
    def forward(self, inputs, adj):
        if self.n_layers == 1:
            x = self.first_layer(inputs, adj)
        else:
            x = self.relu(self.first_layer(inputs, adj))
            if self.n_layers > 2:
                for i, layer in enumerate(self.gc_layers):
                    x = self.relu(layer(x, adj))
                    # x = layer(x, adj)
            # x = self.last_layer(x, adj)
            # 初始化隐藏状态和单元状态（可选）
        h_0 = torch.zeros(2, 11473, 32).to(self.device)  # 初始化隐藏状态
        c_0 = torch.zeros(2, 11473, 32).to(self.device)  # 初始化单元状态
        x = torch.unsqueeze(inputs, dim=2)
        # 前向传播
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        x = h_n[0] + h_n[1]

        # return F.log_softmax(x, dim=1)self.linear(x)
        x=self.MLP(x)
        return F.log_softmax(x, dim=1)


class GCN1(torch.nn.Module):
    def __init__(self, node_features, input_size, num_classes):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(node_features, input_size)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(input_size // 2, input_size // 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(input_size // 4, num_classes))
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        '''
        GCN
        '''
        x = self.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.MLP(x)

        return F.log_softmax(x, dim=1)

