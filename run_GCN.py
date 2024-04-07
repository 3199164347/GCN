#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:13:25 2021

@author: Gong Dongsheng
"""
import copy
import time
import torch
import numpy as np
from GCN import *
from utils import *
from metrics import *
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
"""
pubmed:
n_features = 500
n_classes = 3

cora:
n_features = 1433
n_classes = 7

citeseer:
n_features = 3703
n_classes = 7

cicids:
n_features = 3703
n_classes = 7
"""
dataset = "cicids"
n_features = 34
hidden_dim = 32 # 8 32
dropout = 0.5
n_classes = 6
n_epochs = 1500
early_stop = 10
weight_decay = 5e-4
learning_rate = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device={}'.format(device))
# device = 'cpu'
def l2_reg(model, weight_decay):
    reg = 0.0
    if weight_decay == 0:
        return reg
    for name, parameter in model.first_layer.named_parameters():
        if 'weight' in name:
            reg += weight_decay * (parameter ** 2).sum()
    return reg


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)##load_data自己写，改y_train以后的，分训练集测试集，mask大概也得改
adj = adj.to(device)

features = preprocess_features(features).to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

# features
adj = preprocess_adj(adj, device)

# # features不用稀疏矩阵了，直接改成矩阵。。。
# features = torch.sparse.FloatTensor(
#     torch.LongTensor(features[0].transpose()),
#     torch.FloatTensor(features[1]),
#     torch.Size(features[2])
# )
# features = features.to_dense()
# # 邻接矩阵，稀疏表示
# adj = torch.sparse.FloatTensor(
#     torch.LongTensor(adj[0].transpose()),
#     torch.FloatTensor(adj[1]),
#     torch.Size(adj[2])
# )
# y_train onehot变成label，适应pytorch api...
y_train = torch.FloatTensor(y_train).argmax(dim=1).to(device)
y_val = torch.FloatTensor(y_val).argmax(dim=1).to(device)
y_test = torch.FloatTensor(y_test).argmax(dim=1).to(device)

test_accls = []
with open("./output/info_m4d7.txt", "w") as file:
    for n_layers in range(1,4):
        print("{} layers----------------------------------------".format(n_layers), file=file)
        print("{} layers----------------------------------------".format(n_layers))
        model = GCN(n_layers=n_layers, n_features=n_features, hidden_dim=hidden_dim, dropout=dropout,
                    n_classes=n_classes, device=device).to(device)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_val_loss = 0.0
        best_epoch = 0
        train_accls = []
        y_train_loss = []
        val_accls = []
        for epoch in range(n_epochs):
            t = time.time()
            # train
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss = masked_softmax_cross_entropy(loss_func, output, y_train, copy.deepcopy(train_mask))
            y_train_loss.append(loss.item())
            # reg = l2_reg(model, weight_decay)
            # loss += (reg / len(train_mask))
            train_loss = loss.item()
            print("n_layers={},epoch={}, loss={}".format(n_layers, epoch, loss))
            loss.backward()  # 计算gradient
            optimizer.step()  # 更新parameter

            if (epoch + 1) % 5 == 0:
                # validation
                model.eval()
                t_test = time.time()
                output = model(features, adj)
                val_loss = masked_softmax_cross_entropy(loss_func, output, y_val, copy.deepcopy(val_mask)).item()
                train_acc = masked_accuracy(output, y_train, copy.deepcopy(train_mask))
                val_acc = masked_accuracy(output, y_val, copy.deepcopy(val_mask))
                train_accls.append(train_acc)
                val_accls.append(val_acc)
                train_time = t_test - t
                val_time = time.time() - t_test

                # print("epoch {} | train ACC {} % | val ACC {} % | train loss {} | val loss {}".format(
                #     epoch + 1,
                #     np.round(train_acc * 100, 6),
                #     np.round(val_acc * 100, 6),
                #     np.round(train_loss, 6),
                #     np.round(val_loss, 6)
                # ), file=file)
                print("epoch {} | train ACC {} % | val ACC {} % | train loss {} | val loss {}".format(
                    epoch + 1,
                    np.round(train_acc * 100, 6),
                    np.round(val_acc * 100, 6),
                    np.round(train_loss, 6),
                    np.round(val_loss, 6)
                ))

            '''
            if epoch >= best_epoch + early_stop:
                break
            '''
            if (epoch + 1) % 10 == 0:
                model.eval()
                output = model(features, adj)
                test_acc = masked_accuracy(output, y_test, copy.deepcopy(test_mask))
                print("test  ACC: {} %".format(np.round(test_acc * 100, 5)), file=file)
                print("test  ACC: {} %".format(np.round(test_acc * 100, 5)))
                recall_score = get_recall(output, y_test)
                print("recall score: {}".format(recall_score), file=file)
                print("recall score: {}".format(recall_score))
                # F1_score = get_f1_score(output, y_test)
                # print("F1 score: {}".format(F1_score), file=file)
                # print("F1 score: {}".format(F1_score))
                # precision_score = get_precision_score(output, y_test)
                # print("precision score: {}".format(precision_score), file=file)
                # print("precision score: {}".format(precision_score))

            # if epoch + 1 == n_epochs:
            #     test_accls.append(test_acc)

        """
        model.eval()
        output = model(features, adj)
        test_acc = masked_accuracy(output, y_test, test_mask)
        test_accls.append(test_acc)
        print("test  ACC: {} %".format(np.round(test_acc * 100, 5)))
        """

        recall_score = get_recall(output, y_test)
        print("recall score: {}".format(recall_score), file=file)
        print("recall score: {}".format(recall_score))
        F1_score = get_f1_score(output, y_test)
        print("F1 score: {}".format(F1_score), file=file)
        print("F1 score: {}".format(F1_score))
        precision_score = get_precision_score(output, y_test)
        print("precision score: {}".format(precision_score), file=file)
        print("precision score: {}".format(precision_score))

        plt.figure(figsize=(10, 8))
        plt.title("Dataset: {}".format(dataset))
        plt.plot(list(range(5, n_epochs + 5, 5)), val_accls, label="validation", linewidth=3)
        plt.plot(list(range(5, n_epochs + 5, 5)), train_accls, label="train", linewidth=3)
        plt.xlabel("n epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("output/" + str(n_layers) + "_layers_m4d7_1.png")
        plt.show()

        plt.figure()
        # 去除顶部和右边框框
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('iters')  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        x_train_loss = range(len(y_train_loss))

        plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
        plt.legend()
        plt.title('Loss curve')
        plt.savefig("output/"+ str(n_layers) + "_layers_loss_1.png")
        plt.show()



