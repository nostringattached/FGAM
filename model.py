#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGAM(nn.Module):
    def __init__(self, n_classes, dim_modifiable, dim_unmodifiable, n_embedding, n_hid, batch_norm=False):
        super(FGAM, self).__init__()

        self.n_classes = n_classes
        self.dim_modifiable = dim_modifiable
        self.dim_unmodifiable = dim_unmodifiable
        self.n_embedding = n_embedding
        self.n_hid = n_hid
        self.batch_norm = batch_norm
        self.embedding_layers = nn.ModuleList()

        for i in range(self.dim_modifiable):
            branch = nn.ModuleList()
            branch.append(nn.Linear(1, self.n_embedding))
            for _ in range(n_hid):
                branch.append(nn.Linear(self.n_embedding, self.n_embedding))
                if self.batch_norm:
                    branch.append(nn.BatchNorm1d(self.n_embedding))
            branch.append(nn.Linear(self.n_embedding, 128))
            branch.append(nn.Linear(128, 1))
            self.embedding_layers.append(branch)
        self.linear = nn.Linear(self.dim_modifiable, self.n_classes)

        # for logits[0]
        self.weights_module = nn.ModuleList()
        self.weights_module.append(nn.Linear(self.dim_unmodifiable, 128))
        self.weights_module.append(nn.Linear(128, 128))
        self.weight = nn.Linear(128, self.dim_modifiable*1)
        self.bias = nn.Linear(128, 1)

        # for logits[1]
        self.weights_module2 = nn.ModuleList()
        self.weights_module2.append(nn.Linear(self.dim_unmodifiable, 128))
        self.weights_module2.append(nn.Linear(128, 128))
        self.weight2 = nn.Linear(128, self.dim_modifiable*1)
        self.bias2 = nn.Linear(128, 1)



    def forward(self, unchangeables, changeables):
        res = []
        for i in range(self.dim_modifiable):
            x = changeables[i]
            for j, op in enumerate(self.embedding_layers[i]):
                # no relu before batchNorm
                if (j < (len(self.embedding_layers[i])-1) and
                        (self.embedding_layers[i][j+1]._get_name() ==
                         'BatchNorm1d')) or\
                        (j == (len(self.embedding_layers[i])-1)):
                    x = op(x)
                else:
                    x = F.relu(op(x))
            res.append(x)
        x = torch.cat(res, 1)
        w = unchangeables
        w2 = unchangeables
        for i, op in enumerate(self.weights_module):
            w = F.relu(op(w))
        weight = self.weight(w)
        bias = self.bias(w)
        logits0 = torch.sum(x * weight, dim=1, keepdim=True) + bias

        for i, op in enumerate(self.weights_module2):
            w2 = F.relu(op(w2))
        weight2 = self.weight2(w2)
        bias2 = self.bias2(w2)
        logits1 = torch.sum(x * weight2, dim=1, keepdim=True) + bias2

        logits = torch.cat((logits0, logits1), 1)
        return logits


