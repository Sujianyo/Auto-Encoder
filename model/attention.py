import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    # [b, c, h, w] => [b, h*w, c] => [b, c, h, w]
    def forward(self, query, key):
        b1, n1, h, w = query.shape
        query = query.view(b1, n1, h*w).transpose(1, 2)
        key = key.view(b1, n1, h*w).transpose(1, 2)
        b, n, _ = query.shape  # batch, sequence_length, embedding_dim

        query = self.norm(query)
        key = self.norm(key)
        value = self.norm(key)

        q_ = self.to_q(query)
        k_ = self.to_k(key)
        v_ = self.to_v(value)

        # reshape (b, n, heads * dim_head) -> (b, heads, n, dim_head)
        q = q_.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k_.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = v_.view(b, n, self.heads, self.dim_head).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (b, heads, n, dim_head)

        # reshape back to (b, n, heads * dim_head)
        out = out.transpose(1, 2).contiguous().view(b, n, self.heads * self.dim_head)
        out = self.to_out(out)
        return out.transpose(1, 2).view(b1, n1, h, w)
# import numpy as np
# m = MultiHeadAttention(80, heads=4, dim_head=20)
# # print(m.to_q.weight.shape)
# # m.to_q.weight = nn.Parameter(torch.ones_like(m.to_q.weight)/512)
# # print(m.to_q(torch.tensor(np.linspace(512, 512)))[:, 1])

# # x = torch.rand(1, 1000, 512)  # (batch, seq_len, dim)
# x = torch.rand(1, 80, 100, 100)
# print(m(x, x).shape)



## Each channel is feature, and each pixel is a sample
## eg. [n, c, h, w], has h*w samples
class Axial_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0):
        super().__init__()
        in_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.q = nn.Linear(dim, in_dim)
        self.k = nn.Linear(dim, in_dim)
        self.v = nn.Linear(dim, in_dim)

        self.to_out = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, query, key):
        b, n, _ = query.shape  # batch, sequence_length, embedding_dim

        query = self.norm(query)
        key = self.norm(key)

        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(key)

        # reshape (b, n, heads * dim_head) -> (b, heads, n, dim_head)
        q = q.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, n, self.heads, self.dim_head).transpose(1, 2)

        dots = torch.einsum('bhnw', q, k)* self.scale


        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (b, heads, n, dim_head)

        # reshape back to (b, n, heads * dim_head)
        out = out.transpose(1, 2).contiguous().view(b, n, self.heads * self.dim_head)

        return self.to_out(out)