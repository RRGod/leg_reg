import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Reduce
import torch.nn.functional as F

# helpers

def exists(val):
    return val is not None

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv3d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv3d(dim * mult, dim, 1)
    )

# MHRAs (multi-head relation aggregators)

class LocalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        local_aggr_kernel = 5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        # they use batchnorm for the local MHRA instead of layer norm
        self.norm = nn.BatchNorm3d(dim)

        # only values, as the attention matrix is taking care of by a convolution
        self.to_v = nn.Conv3d(dim, inner_dim, 1, bias = False)

        # this should be equivalent to aggregating by an attention matrix parameterized as a function of the relative positions across each axis
        self.rel_pos = nn.Conv3d(heads, heads, local_aggr_kernel, padding = local_aggr_kernel // 2, groups = heads)

        # combine out across all the heads
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        b, c, *_, h = *x.shape, self.heads

        # to values
        v = self.to_v(x)

        # split out heads
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h = h)

        # aggregate by relative positions
        out = self.rel_pos(v)

        # combine heads
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b = b)
        return self.to_out(out)

class GlobalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        shape, h = x.shape, self.heads

        x = rearrange(x, 'b c ... -> b c (...)')

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n', h = h)

        out = self.to_out(out)
        return out.view(*shape)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mhsa_type = 'g',
        local_aggr_kernel = 5,
        dim_head = 64,
        ff_mult = 4,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mhsa_type == 'l':
                attn = LocalMHRA(dim, heads = heads, dim_head = dim_head, local_aggr_kernel = local_aggr_kernel)
            elif mhsa_type == 'g':
                attn = GlobalMHRA(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv3d(dim, dim, 3, padding = 1),
                attn,
                FeedForward(dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    def forward(self, x):
        for dpe, attn, ff in self.layers:
            x = dpe(x) + x
            x = attn(x) + x
            x = ff(x) + x

        return x

# main class

class Uniformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dims = (64, 128, 256, 512),
        depths = (3, 4, 8, 3),
        mhsa_types = ('l', 'l', 'g', 'g'),
        local_aggr_kernel = 5,
        channels = 3,
        ff_mult = 4,
        dim_head = 64,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        hidden_dim = 8
        super().__init__()
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv3d(channels, init_dim, (3, 4, 4), stride = (2, 4, 4), padding = (1, 0, 0))

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))
        mhsa_types = tuple(map(lambda t: t.lower(), mhsa_types))

        self.stages = nn.ModuleList([])
        self.bin_num = [1, 2, 4, 8]  # 水平条的尺度
        # 全连接层，hidden_dim = 256
        # sum(self.bin_num) * 2, 128, hidden_dim = 31 * 2, 128, 256 = 62, 128, 256
        # 反向传播过程中，利用该全连接矩阵与最后特征做tensor乘法以实现维度扩展，注意实现参数初始化
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num), 5, hidden_dim)))])

        for ind, (depth, mhsa_type) in enumerate(zip(depths, mhsa_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Transformer(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mhsa_type = mhsa_type,
                    ff_mult = ff_mult,
                    ff_dropout = ff_dropout,
                    attn_dropout = attn_dropout
                ),
                nn.Sequential(
                    nn.Conv3d(stage_dim, dims[ind + 1], (1, 2, 2), stride = (1, 2, 2)),
                    LayerNorm(dims[ind + 1]),
                ) if not is_last else None
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b c t h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, video):
        video = video.type(torch.cuda.FloatTensor)
        x = self.to_tokens(video)

        for transformer, conv in self.stages:
            x = transformer(x)

            if exists(conv):
                x = conv(x)

        return self.to_logits(self.HPM_3D(x))

    def HPM(self,input):
        feature = list()
        input = F.pad(input, (0, 13, 0, 0), mode='constant', value=0)
        n, c, h, w = input.size()
        for num_bin in self.bin_num:
            z = input.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()

        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 0, 2).contiguous()
        return feature

    def HPM_3D(self,input):
        split_tensors = torch.split(input, 1, dim=1)
        split_tensors = [t.squeeze(1) for t in split_tensors]
        stacked_tensors = []
        for split_tensor in split_tensors:
            stacked_tensor = self.HPM(split_tensor)
            stacked_tensors.append(stacked_tensor)
        # 假设 split_tensors 是包含512个形状为 (8,5,3,3) 的张量的列表
        ulti_tensors = torch.stack(split_tensors, dim=1)
        return ulti_tensors


if __name__ == "__main__":

    inputs = torch.rand(8, 3, 10, 112, 112)
    net = Uniformer(num_classes=7)

    outputs = net.forward(inputs)
    print(outputs.size())