#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 17:16:28 2021

@author: Gong Dongsheng
"""
import numpy as np
import torch
from sklearn import metrics


def masked_softmax_cross_entropy(org_loss_func, preds, labels, mask):
    # preds_=preds.argmax(dim=1)
    loss = org_loss_func(preds, labels)
    _mask = mask
    _mask /= _mask.sum()
    loss *= _mask
    # return loss.mean()
    return loss.sum()


def masked_accuracy(preds, labels, mask):
    acc = torch.eq(preds.argmax(1), labels).float()
    # _mask = torch.FloatTensor(mask)
    _mask = mask
    _mask /= _mask.sum()
    acc *= _mask
    return acc.sum().item()

def get_recall(preds, labels):
    label = [0, 1, 2, 3, 4, 5]
    y_true = labels.cpu()
    y_pred = torch.FloatTensor(preds.cpu()).argmax(dim=1)
    return metrics.recall_score(y_pred, y_true, average=None)

def get_f1_score(preds, labels):
    label = [0, 1, 2, 3, 4, 5]
    y_true = labels[-10000:].cpu()
    y_pred = torch.FloatTensor(preds[-10000:].cpu()).argmax(dim=1)
    return metrics.f1_score(y_pred, y_true, labels=label, average=None)

def get_precision_score(preds, labels):
    label = [0, 1, 2, 3, 4, 5]
    y_true = labels[-10000:].cpu()
    y_pred = torch.FloatTensor(preds[-10000:].cpu()).argmax(dim=1)
    return metrics.precision_score(y_pred, y_true, labels=label, average=None)