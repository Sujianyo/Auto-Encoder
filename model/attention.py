import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
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
class Attention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__(embed_dim, num_heads, dropout=0.0, bias=True,
                                                         add_bias_kv=False, add_zero_attn=False,
                                                         kdim=None, vdim=None)

    def forward(self, query, key, value=None, pos_enc=None, pos_indexes=None):
        # print(query.shape, key.shape, value.shape, pos_enc.shape, pos_indexes.shape)
        w, bsz, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # project to get qkv
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # cross-attention
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        # project to find q_r, k_r
        if pos_enc is not None:
            # reshape pos_enc
            pos_enc = torch.index_select(pos_enc, 0, pos_indexes).view(w, w,
                                                                       -1)  # 2W-1xC -> WW'xC -> WxW'xC
            # compute k_r, q_r
            _start = 0
            _end = 2 * embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            _b = self.in_proj_bias[_start:_end]
            q_r, k_r = F.linear(pos_enc, _w, _b).chunk(2, dim=-1)  # WxW'xC
        else:
            q_r = None
            k_r = None

        # scale query
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        if q_r is not None:
            q_r = q_r * scaling

        # reshape
        q = q.contiguous().view(w, bsz, self.num_heads, head_dim)  # WxNxExC
        if k is not None:
            k = k.contiguous().view(-1, bsz, self.num_heads, head_dim)
        if v is not None:
            v = v.contiguous().view(-1, bsz, self.num_heads, head_dim)

        if q_r is not None:
            q_r = q_r.contiguous().view(w, w, self.num_heads, head_dim)  # WxW'xExC
        if k_r is not None:
            k_r = k_r.contiguous().view(w, w, self.num_heads, head_dim)

        # compute attn weight
        attn_feat = torch.einsum('wnec,vnec->newv', q, k)  # NxExWxW'

        # add positional terms
        if pos_enc is not None:
            # 0.3 s
            attn_feat_pos = torch.einsum('wnec,wvec->newv', q, k_r)  # NxExWxW'
            attn_pos_feat = torch.einsum('vnec,wvec->newv', k, q_r)  # NxExWxW'

            # 0.1 s
            attn = attn_feat + attn_feat_pos + attn_pos_feat
        else:
            attn = attn_feat

        assert list(attn.size()) == [bsz, self.num_heads, w, w]


        # softmax
        attn = F.softmax(attn, dim=-1)

        # compute v, equivalent to einsum('',attn,v),
        # need to do this because apex does not support einsum when precision is mixed
        v_o = torch.bmm(attn.view(bsz * self.num_heads, w, w),
                        v.permute(1, 2, 0, 3).view(bsz * self.num_heads, w, head_dim))  # NxExWxW', W'xNxExC -> NExWxC
        assert list(v_o.size()) == [bsz * self.num_heads, w, head_dim]
        v_o = v_o.reshape(bsz, self.num_heads, w, head_dim).permute(2, 0, 1, 3).reshape(w, bsz, embed_dim)
        v_o = F.linear(v_o, self.out_proj.weight, self.out_proj.bias)

        return v_o
class Axial_Attention(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__( )
        self.self_attn1 = Attention(hidden_dim, nhead)
        self.self_attn2 = Attention(hidden_dim, nhead)
        self.norm1 = nn.LayerNorm(hidden_dim)
    def forward(self, feat,
                pos = None,
                pos_indexes = None,
                pos_y = None,
                pos_indexes_y = None,
                ):
        """
        :param feat: image feature [W,2HN,C]
        :param pos: pos encoding [2W-1,HN,C]
        :param pos_indexes: indexes to slice pos encoding [W,W]
        :return: updated image feature
        """
        feat2 = self.norm1(feat)

        w, h2, c=feat2.shape
        feat2=feat2.reshape(w,2,h2//2,c).permute(2,1,0,3).reshape(h2//2,2*w,c)
        # print(pos_indexes_y.shape, pos_y)
        feat2 = self.self_attn2(query=feat2, key=feat2, value=feat2, pos_enc=pos_y,
                                               pos_indexes=pos_indexes_y)
        
        feat2=feat2.reshape(h2//2,2,w,c).permute(2,1,0,3).reshape(w,h2,c)
        feat2 = self.self_attn1(query=feat2, key=feat2, value=feat2, pos_enc=pos,
                                               pos_indexes=pos_indexes)
        # bachsize(N)=1 [W,2HN,C]->[H,2WN,C]
        # torch.save(attn_weight, 'self_attn_' + str(layer_idx) + '.dat')
        feat = feat + feat2
        # print(feat.shape)
        return feat
    
class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer  test
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = Attention(hidden_dim, nhead)
        self.cross_attn1 = Attention(hidden_dim, nhead)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.merge=nn.Sequential(nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                 nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,stride=1,padding=1))
        #self.merge1=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3,stride=1,padding=1)
    def forward(self, feat_left: Tensor, feat_right: Tensor, 
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None
                ):
        """
        :param feat_left: left image feature, [W,HN,C]
        :param feat_right: right image feature, [W,HN,C]
        :param pos: pos encoding, [2W-1,HN,C]
        :param pos_indexes: indexes to slicer pos encoding [W,W]
        :param last_layer: Boolean indicating if the current layer is the last layer
        :return: update image feature and attention weight
        """
        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm1(feat_right)
        # torch.save(torch.cat([feat_left_2, feat_right_2], dim=1), 'feat_cross_attn_input_' + str(layer_idx) + '.dat')

        # update right features
        if pos is not None:
            pos_flipped = torch.flip(pos, [0])
        else:
            pos_flipped = pos

        feat_right_2 = self.cross_attn(query=feat_right_2, key=feat_left_2, value=feat_left_2, pos_enc=pos_flipped,
                                       pos_indexes=pos_indexes)

        feat_right = feat_right + feat_right_2
        # update left features
        # use attn mask for last layer

        # normalize again the updated right features
        feat_right_2 = self.norm1(feat_right)
        feat_left_2 = self.cross_attn(query=feat_left_2, key=feat_right_2, value=feat_right_2, pos_enc=pos, pos_indexes=pos_indexes)

        # torch.save(attn_weight, 'cross_attn_' + str(layer_idx) + '.dat')

        feat_left = feat_left + feat_left_2
        # concat features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC
        # feat_new=self.merge(feat.permute(2,1,0).unsqueeze(0)).squeeze().permute(2,1,0)
        # print(feat_new.shape)
        return feat
import math
class PositionEncodingSine1DRelative(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @torch.no_grad()
    def forward(self, inputs):
        """
        :param inputs: NestedTensor
        :return: pos encoding [2W-1,C]
        """
        x = inputs
        bs, _, h, w = x.size()
        #1/4
        w = math.ceil(w)
        h = math.ceil(h)

        # populate all possible relative distances
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=x.device)
        y_embed = torch.linspace(h - 1, -h + 1, 2 * h - 1, dtype=torch.float32, device=x.device)

        if self.normalize:
            x_embed = x_embed * self.scale
            y_embed = y_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t  # 2W-1xC
        pos_y = y_embed[:, None] / dim_t  # 2H-1xC

        # interleave cos and sin instead of concatenate
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)   # 2H-1xC

        return pos_x, pos_y
import copy
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class Transformer(nn.Module):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    def __init__(self, hidden_dim: int = 128, nhead: int = 8, num_attn_layers: int = 6):
        super().__init__()

        self_attn_layer = Axial_Attention(hidden_dim, nhead)
        self.self_attn_layers = get_clones(self_attn_layer, num_attn_layers)

        cross_attn_layer = TransformerCrossAttnLayer(hidden_dim, nhead)
        self.cross_attn_layers = get_clones(cross_attn_layer, num_attn_layers)

        self.norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers

    def _alternating_attn(self, feat: torch.Tensor,
                        pos_enc: torch.Tensor, pos_indexes: Tensor ,
                        pos_enc_y: torch.Tensor, pos_indexes_y: Tensor ,
                        hn: int):
        """
        Alternate self and cross attention with gradient checkpointing to save memory

        :param feat: image feature concatenated from left and right, [W,2HN,C]
        :param feat1: small image feature concatenated from left and right, [W,2HN,C]
        :param pos_enc: positional encoding, [W,HN,C]
        :param pos_indexes1: indexes to slice positional encoding, [W,HN,C]
        :param pos_enc_y: positional encoding along y axis
        :param pos_indexes_y: indexes to slice positional encoding along y axis
        :param hn: size of HN
        :return: attention weight [N,H,W,W]
        """

        global layer_idx
        # alternating
        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            layer_idx = idx

            # checkpoint self attn
            def create_custom_self_attn(module):
                def custom_self_attn(*inputs):
                    return module(*inputs)

                return custom_self_attn

            # if visualize == True:
            #     torch.save(feat,'feat_self_attn_input_' + str(layer_idx) + '.dat')
            feat = checkpoint(create_custom_self_attn(self_attn), feat,
                               pos_enc, pos_indexes ,
                               pos_enc_y, pos_indexes_y )

            def create_custom_cross_attn(module):
                def custom_cross_attn(*inputs):
                    return module(*inputs)
                return custom_cross_attn
            # if visualize==True:
            #     torch.save(feat,'feat_cross_attn_input_' + str(layer_idx) + '.dat')
            feat = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:],pos_enc,
                                            pos_indexes)
        layer_idx = 0
        return feat
    def forward(self, feat1, feat2, pos_enc=None, pos_enc_y=None):
        bs, c, h, w = feat1.shape # n, c, h, w -> c, w, h, n -> c, w, h*n -> w, h*n, c

        feat_left = feat1.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        feat_right = feat2.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        if pos_enc is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes = None
        if pos_enc_y is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r1 = torch.linspace(h - 1, 0, h).view(h, 1).to(feat_left.device)
                indexes_c1 = torch.linspace(0, h - 1, h).view(1, h).to(feat_left.device)
                pos_indexes_y = (indexes_r1 + indexes_c1).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes_y = None
        feat = torch.cat([feat_left, feat_right], dim=1)
        # print(feat.shape)
        x = self._alternating_attn(feat,
                        pos_enc, pos_indexes ,
                        pos_enc_y, pos_indexes_y ,
                        h)
        return torch.cat([x[:, :h, :], x[:, h:, :]], dim=2).permute(2, 0, 1).unsqueeze(0)

# m = Transformer(num_attn_layers=2)
# pos = PositionEncodingSine1DRelative(num_pos_feats=128)
# x = torch.rand(1, 128, 60, 100)
# a, b = pos(x)
# print(m(x, x, a, b).size())