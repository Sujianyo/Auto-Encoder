import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., patch_kernel=3, pos_abs=True):
        super().__init__()
        inner_dim = dim * patch_kernel**2
        project_out = not (heads == 1)
        
        self.patch_kernel = patch_kernel

        self.heads = heads
        self.dim_head = inner_dim//heads
        self.scale = self.dim_head ** -0.5

        self.norm = nn.LayerNorm(inner_dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        # self.to_k = nn.Linear(inner_dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(inner_dim, inner_dim, bias=False)
        ## inner_dim = 64*7*7 = 3136
        ## para_num = inner_dim * inner_dim = 3136**2
        self.to_q = nn.Conv2d(dim, inner_dim, kernel_size=patch_kernel, stride=patch_kernel)
        self.to_k = nn.Conv2d(dim, inner_dim, kernel_size=patch_kernel, stride=patch_kernel)
        self.to_v = nn.Conv2d(dim, inner_dim, kernel_size=patch_kernel, stride=patch_kernel)
        ## dim * inner_dim * patch**2 = 3136 * 64 * 49 = 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.pos = Position(inner_dim, pos_abs)
        # self.pos_embed = Position()
    def patch(self, x):
        x = F.unfold(x, kernel_size=self.patch_kernel, stride=self.patch_kernel)
        return x.transpose(1, 2)
    def unpatch(self, x, output_size):
        x = x.transpose(1, 2)
        return F.fold(x, output_size=output_size, kernel_size=self.patch_kernel, stride=self.patch_kernel)    
    def forward(self, query, key=None):
    # def forward(self, query, key):
        # cross attention
        _, _, h, w = query.shape
        
        # query = self.patch(query)
        # key = self.patch(key)

        # b, n, _ = query.shape  # batch, sequence_length, embedding_dim
        # query = self.norm(query)
        # key = self.norm(key)
        # value = self.norm(key)
        res_q = self.patch(query)
        q_ = self.to_q(query).flatten(-2).transpose(1, 2)
        if key is not None:
            k_ = self.to_k(key).flatten(-2).transpose(1, 2)
            v_ = self.to_v(key).flatten(-2).transpose(1, 2)
        else:
            k_ = self.to_k(query).flatten(-2).transpose(1, 2)
            v_ = self.to_v(query).flatten(-2).transpose(1, 2)

        b, n, _ = q_.shape
        ## Position embedding
        pos = self.pos(n).to(q_.device)
        # pos = 0
        q_ += pos
        k_ += pos
        # q_ = self.norm(q_)
        # k_ = self.norm(k_)
        # v_ = self.norm(v_)



        # reshape (b, n, heads * dim_head) -> (b, heads, n, dim_head)
        q = q_.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k_.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = v_.view(b, n, self.heads, self.dim_head).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(2, 3)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (b, heads, n, dim_head)

        # reshape back to (b, n, heads * dim_head)
        out = out.transpose(1, 2).contiguous().view(b, n, self.heads * self.dim_head)
        out = self.to_out(out)
        out = self.norm(out + res_q)
        return self.unpatch(out, (h, w))

class Position(nn.Module):
    def __init__(self, dim, abs=True):
        super().__init__()
        self.dim = dim
        self.cache = {}
    @torch.no_grad
    def forward(self, n_samples):
        # if n_samples in self.cache:
        #     return self.cache[n_samples]

        position = torch.arange(n_samples).unsqueeze(1)
        positional_encoding = torch.zeros(1, n_samples, self.dim)
        _2i = torch.arange(0, self.dim, step=2).float()
        positional_encoding[0, :, 0::2] = torch.sin(position / (10000 ** (_2i / self.dim)))
        positional_encoding[0, :, 1::2] = torch.cos(position / (10000 ** (_2i / self.dim)))

        # self.cache[n_samples] = positional_encoding
        return positional_encoding

# @torch.no_grad
# class Position(nn.Module):
#     def __init__(self, dim, abs=True):
#         super().__init__()
#         self.dim = dim
#         self.cache = {}
#     @torch.no_grad
#     def forward(self, n_samples):
#         if n_samples in self.cache:
#             return self.cache[n_samples]

#         position = torch.arange(n_samples).unsqueeze(1)
#         positional_encoding = torch.zeros(1, n_samples, self.dim)
#         _2i = torch.arange(0, self.dim, step=2).float()
#         positional_encoding[0, :, 0::2] = torch.sin(position / (10000 ** (_2i / self.dim)))
#         positional_encoding[0, :, 1::2] = torch.cos(position / (10000 ** (_2i / self.dim)))

#         self.cache[n_samples] = positional_encoding
#         return positional_encoding




    # [b, c, h, w] => [b, h*w, c] => [b, c, h, w]
    # def forward(self, query):
    #     # self attention
    #     _, _, h, w = query.shape
    #     query = self.patch(query)
    #     b, n, c = query.shape
    #     query = self.norm(query)

    #     q_ = self.to_q(query)
    #     k_ = self.to_k(query)
    #     v_ = self.to_v(query)

    #     q = q_.view(b, n, self.heads, self.dim_head).transpose(1, 2)
    #     k = k_.view(b, n, self.heads, self.dim_head).transpose(1, 2)
    #     v = v_.view(b, n, self.heads, self.dim_head).transpose(1, 2)

    #     dots = torch.matmul(q, k.transpose(2, 3)) * self.scale

    #     attn = self.attend(dots)
    #     attn = self.dropout(attn)

    #     out = torch.matmul(attn, v)  # (b, heads, n, dim_head)

    #     # reshape back to (b, n, heads * dim_head)
    #     out = out.transpose(1, 2).contiguous().view(b, n, self.heads * self.dim_head)
    #     out = self.to_out(out)

    #     return self.unpatch(out, (h, w))



import torch

# torch.cuda.reset_peak_memory_stats()

# m = MultiHeadAttention(dim=64, heads=8, dropout=0.).to('cpu')
# x = torch.rand(1, 64, 256, 256).to('cpu')
# print(m(x).shape)

# max_mem = torch.cuda.max_memory_allocated()

# print(f"Max memery: {max_mem / 1024**2:.2f} MB")


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


    def forward(self, query, key = None):
        b, n, h, w = query.shape  # batch, sequence_length, embedding_dim
        query = query.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        

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