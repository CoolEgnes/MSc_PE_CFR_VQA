"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from src.bc import BCNet

class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        # torch.nn.utils.weight_norm(module, name='weight', dim=0): Apply weight normalization to a parameter in the given module.
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=1), \
            name='h_mat', dim=None)  # apply the weigth_norm to the self.h_mat layer in the BCNet

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)  # Tensor: (64, 12, 1024), device: cuda, v_num: 12
        q_num = q.size(1)  # Tensor: (64, 7, 600), device: cuda, q_num: 7
        # matmul tensor of (v,q)
        logits = self.logits(v, q)  # b x g x v x q

        # if v_mask:
        #     mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
        #     logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits
