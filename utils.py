#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:40:30 2021

@author: tkipf
https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""
import csv
import random

import pandas as pd
import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh
import torch.nn.functional as F
import joblib


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return mask


def load_data(dataset_str):
    # adj = torch.load('./normalized_adj1.pt')
    adj = torch.load('./origin/adj_extend.pt')
    features = torch.load('./origin/features_extend.pt')
    features[features == float('inf')] = 0.
    features = torch.nan_to_num(features)
    labels = joblib.load('./origin/label_extend.pkl')
    labels = np.array([[0 if j != labels[i] else 1 for j in range(6)] for i in range(len(labels))])
    all_dix = [i for i in range(len(features))]
    idx_train =all_dix[:int(len(all_dix)*0.8)]  # 有标签的
    idx_val = all_dix[int(len(all_dix)*0.6):int(len(all_dix)*0.8)]
    idx_test = all_dix[int(len(all_dix)*0.8):]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask.bool(), :] = labels[train_mask.bool(), :]
    y_val[val_mask.bool(), :] = labels[val_mask.bool(), :]
    y_test[test_mask.bool(), :] = labels[test_mask.bool(), :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def getLabels(line, col):
    labels = np.zeros((line, col))
    data_frame = pd.read_csv('input_1w_modified.csv')
    for index, row in data_frame.iterrows():
        if row[84] == 'BENIGN':
            labels[index][0] = 1
        elif row[84][0:3] == 'DoS' or row[84][0:3] == 'Dos':
            labels[index][1] = 1
        elif row[84] == 'DDoS':
            labels[index][2] = 1
        elif row[84] == 'PortScan':
            labels[index][3] = 1
        elif row[84] == 'Bot':
            labels[index][4] = 1
        elif row[84][0:3] == 'Web':
            labels[index][5] = 1
    return labels

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    解释一下:
        scipy coordinate稀疏矩阵类型转换成tuple
        coords[i]是一个二元组，代表一个坐标
        values[i]代表sparse_matrix[coords[i][0], coords[i][1]]的value
        shape=(2708, 1433)是稀疏矩阵的维度
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # rowsum = np.array(features.sum(0))  # 每个node所有通道求和。用于feature归一化
    # r_inv = np.power(rowsum, -1).flatten()  # 1除以rowsum，再flatten
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)  # diag(r_inv)
    # features = r_mat_inv.dot(features)  # features * diag(r_inv), 是矩阵乘法，不是element wise
    # return sparse_to_tuple(features)
    # features = features / torch.norm(features, dim=0, keepdim=True)
    # 计算每个节点所有通道的和
    # rowsum = torch.sum(features, dim=0)
    #
    # # 计算 1 / rowsum
    # r_inv = torch.pow(rowsum, -1).flatten()
    #
    # # 处理无穷大的情况
    # r_inv[r_inv == float('inf')] = 0.
    # r_inv[r_inv == float('nan')] = 0.
    #
    # # 构建对角矩阵 1 / rowsum
    # r_mat_inv = torch.diag(r_inv)
    #
    # # 归一化特征矩阵
    # normalized_features = torch.matmul(features, r_mat_inv)
    normalized_tensor = F.normalize(features, p=2, dim=1)
    return normalized_tensor


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    # rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # 计算行之和
    rowsum = torch.sum(adj, dim=1)
    # 计算 D^{-0.5}
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    # 处理无穷大的情况
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.
    # 构建对角矩阵 D^{-0.5}
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # 计算归一化的邻接矩阵
    normalized_adj = torch.matmul(adj, d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    # 转换为稀疏矩阵格式
    # normalized_adj = normalized_adj.to_sparse()
    adj_ = adj.cpu().numpy()
    row_sum_ = np.array(adj_.sum(1))  # 求度矩阵D
    d_inv_sqrt_ = np.power(row_sum_, -0.5).flatten()  # D^-1/2
    d_inv_sqrt_[np.isinf(d_inv_sqrt_)] = 0.  # 将一些计算得到的NAN值赋0值
    d_mat_inv_sqrt_ = np.mat(np.diag(d_inv_sqrt_))  # 将D^-1/2对角化
    gcn_fact = d_mat_inv_sqrt_ * adj_ * d_mat_inv_sqrt_  # 计算D^-1/2AD^-1/2
    return torch.tensor(gcn_fact)


def preprocess_adj(adj, device):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)  # 直接把A+In传入再normalize，就和论文里的A^一样了
    return adj_normalized.to(device)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


# if __name__ == "__main__":
    # adj = torch.load('adj1.pt')
    # rowsum = torch.sum(adj, dim=1)
    # # 计算 D^{-0.5}
    # d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    # # 处理无穷大的情况
    # d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.
    # # 构建对角矩阵 D^{-0.5}
    # d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # # 计算归一化的邻接矩阵
    # normalized_adj = torch.matmul(adj, d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    # # 转换为稀疏矩阵格式
    # # normalized_adj = normalized_adj.to_sparse()
    # torch.save(normalized_adj, 'normalized_adj1.pt')

    # ===============output_txt 存tensor===============
    # adj = [[0]*13061] * 13061
    # adj = torch.ones((13060,13060))
    # with open("adj_matrix.txt", "r") as file:
    #     lines = file.readlines()
    #     for j, line in enumerate(lines):
    #         print(j)
    #         adj[j] = torch.tensor([float(digit) for digit in line.strip()])
    # torch.save(torch.tensor(adj).clone().detach(), "./adj1.pt")


    # ===============fetures 存tensor===============
    # types = [float, float, float, float, float, float, float, float, float]
    #
    # with open('attribute_output.csv', 'r') as file:
    #     reader = csv.reader(file)
    #     # 跳过标题行
    #     headers = next(reader)
    #     features = []
    #     for row in reader:
    #         for i, value in enumerate(row):
    #             # 使用types中的类型转换数据
    #             if torch.tensor(types[i](value)) == torch.inf or torch.tensor(types[i](value)) == torch.nan:
    #                 value=0
    #             row[i] = types[i](value) if i < len(types) else value
    #         # 处理转换后的数据
    #         features.append(row)
    # file_path = "features1.pt"
    #
    # features = torch.tensor(features)
    # torch.save(features.clone().detach(), file_path)
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")
    