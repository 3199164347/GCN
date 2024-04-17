#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 13:06:46 2021

@author: Gong Dongsheng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        h: (N, in_features)
        adj: sparse matrix with shape (N, N)
        '''
        Wh = torch.mm(h, self.W)  # (N, out_features)
        Wh1 = torch.mm(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.mm(Wh, self.a[self.out_features:, :])  # (N, 1)

        # Wh1 + Wh2.T 是N*N矩阵，第i行第j列是Wh1[i]+Wh2[j]
        # 那么Wh1 + Wh2.T的第i行第j列刚好就是文中的a^T*[Whi||Whj]
        # 代表着节点i对节点j的attention
        e = self.leakyrelu(Wh1 + Wh2.T)  # (N, N)
        padding = (-2 ** 31) * torch.ones_like(e)  # (N, N)
        attention = torch.where(adj > 0, e, padding)  # (N, N)
        attention = F.softmax(attention, dim=1)  # (N, N)
        # attention矩阵第i行第j列代表node_i对node_j的注意力
        # 对注意力权重也做dropout（如果经过mask之后，attention矩阵也许是高度稀疏的，这样做还有必要吗？）
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # (N, out_features)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, device):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.MH = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, concat=False)
        # lstm
        self.lstm = nn.LSTM(1, nhid, 2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(nhid, nclass)
        self.relu = nn.ReLU()
        self.device = device
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(32, nhid),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(nhid, nhid // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(nhid // 2, nclass))

    def forward(self, inputs, adj):
        # x = F.dropout(inputs, self.dropout, training=self.training)    # (N, nfeat)
        x = inputs
        # x = torch.cat([head(x, adj) for head in self.MH], dim=1)  # (N, nheads*nhid)
        # x = F.dropout(x, self.dropout, training=self.training)    # (N, nheads*nhid)
        # x = F.elu(self.out_att(x, adj))
        # lstm+mlp
        h_0 = torch.zeros(4, 6000, 32).to(self.device)  # 初始化隐藏状态
        c_0 = torch.zeros(4, 6000, 32).to(self.device)  # 初始化单元状态
        x = torch.unsqueeze(x, dim=2)
        # 前向传播
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        x = h_n[0] + h_n[1]

        x = self.MLP(x)
        return F.log_softmax(x, dim=1)
        # return x