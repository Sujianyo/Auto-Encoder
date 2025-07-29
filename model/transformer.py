import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import copy
class Attention(nn.Module):
    def __init__(self, hidden_dim: int = 128, nhead: int = 8, lst=False):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.nhead = nhead
        self.hidden_dim = hidden_dim

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, q, k=None, v=None):
        n, c, w, h = q.shape
        head_dim = self.hidden_dim // self.nhead
        q = q.flatten(-1).permute(0, 2, 1) # n, w*h, c
        if k == None:
            k = q.clone()
            v = q.clone()
        q = self.q(q).view(n, -1, self.nhead, head_dim)
        k = self.k(k).view(n, -1, self.nhead, head_dim)
        v = self.v(v).view(n, -1, self.nhead, head_dim)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class Transformer(nn.Module):
    def __init__(self, hidden_dim: int = 128, nhead: int = 8, num_attn_layers: int = 4):
        super().__init__()
        # self.conv = get_clones(nn.Conv2d(hidden_dim*2, hidden_dim, 1), num_attn_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = get_clones(Attention(hidden_dim, nhead), num_attn_layers)

        self.cross_layers = get_clones(Attention(hidden_dim, nhead), num_attn_layers)

        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers
    def forward(self, feat_encoder, feat_decoder):
        # feat_decoder: [N, C, H, W]
        for idx, (self_atten, cross_atten) in enumerate(zip(self.attention, self.cross_layers)):
            feat_encoder = self_atten(feat_encoder)
            feat_encoder = cross_atten(feat_encoder, feat_decoder)

