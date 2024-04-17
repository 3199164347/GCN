#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:13:25 2021

@author: Gong Dongsheng
"""

import time
import os
import torch
import numpy as np
from GAT import *
from utils import *
from metrics import *
import torch.optim as optim
import matplotlib.pyplot as plt

dataset = "cicids"
nfeat = 34
nhid = 32
nheads = 8
nclass = 6
alpha = 0.2
dropout = 0.5
n_epochs = 10000
early_stop = 10
weight_decay = 5e-4
learning_rate = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device={}'.format(device))



def l2_reg(model, weight_decay):
    reg = 0.0
    if weight_decay == 0:
        return reg
    for name, parameter in model.named_parameters():
        reg += weight_decay * (parameter ** 2).sum()
    return reg


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
features = preprocess_features(features).to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

# # features不用稀疏矩阵了，直接改成矩阵。。。
# features = torch.sparse.FloatTensor(
#     torch.LongTensor(features[0].transpose()),
#     torch.FloatTensor(features[1]),
#     torch.Size(features[2])
# )
# features = features.to_dense()
# # 邻接矩阵，稀疏表示
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj + torch.eye(adj.shape[0])*100000)
adj = adj.to(device)
# y_train onehot变成label，适应pytorch api...
y_train = torch.FloatTensor(y_train).argmax(dim=1).to(device)
y_val = torch.FloatTensor(y_val).argmax(dim=1).to(device)
y_test = torch.FloatTensor(y_test).argmax(dim=1).to(device)

test_accls = []
model = GAT(nfeat, nhid, nclass, dropout, alpha, nheads, device).to(device)
loss_func = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = 0.0
best_epoch = 0
train_accls = []
val_accls = []

for epoch in range(n_epochs):
    t = time.time()

    # train
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = masked_softmax_cross_entropy(loss_func, output, y_train, train_mask)
    # reg = l2_reg(model, weight_decay)
    # loss += (reg / len(train_mask))
    train_loss = loss.item()
    loss.backward()  # 计算gradient
    optimizer.step()  # 更新parameter

    # validation
    model.eval()
    t_test = time.time()
    output = model(features, adj)
    val_loss = masked_softmax_cross_entropy(loss_func, output, y_val, val_mask).item()
    train_acc = masked_accuracy(output, y_train, train_mask)
    val_acc = masked_accuracy(output, y_val, val_mask)
    train_accls.append(train_acc)
    val_accls.append(val_acc)
    train_time = t_test - t
    val_time = time.time() - t_test

    print("epoch {} | train ACC {} % | val ACC {} % | train loss {} | val loss {}".format(
        epoch + 1,
        np.round(train_acc * 100, 4),
        np.round(val_acc * 100, 4),
        np.round(train_loss, 4),
        np.round(val_loss, 4)
    ))

    if val_loss > best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
    '''
    if epoch >= best_epoch + early_stop:
        break
    '''

model.eval()
output = model(features, adj)
test_acc = masked_accuracy(output, y_test, test_mask)
test_accls.append(test_acc)
print("test  ACC: {} %".format(np.round(test_acc * 100, 4)))

plt.figure(figsize=(10, 8))
plt.title("Dataset: {}".format(dataset))
plt.plot(range(len(val_accls)), val_accls, label="validation", linewidth=3)
plt.plot(range(len(train_accls)), train_accls, label="train", linewidth=3)
plt.xlabel("n epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
