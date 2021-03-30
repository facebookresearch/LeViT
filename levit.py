import torch
import torch.nn as nn
import itertools
import time
import os

from timm.models.vision_transformer import _cfg, trunc_normal_
from timm.models.registry import register_model


@register_model
def LeViT_128S(num_classes, distillation, pretrained=False):
    net = model_factory(C='128_256_384', D=16, N='4_6_8', X='2_3_4',
                        activation='Hardswish', distillation=distillation, num_classes=num_classes)
    if pretrained:
        net.load_state_dict(torch.load(
            'weights/LeViT-128S/checkpoint.pth')['model'])
    return net


@register_model
def LeViT_128(num_classes, distillation, pretrained=False):
    net = model_factory(C='128_256_384', D=16, N='4_8_12', X='4_4_4',
                        activation='Hardswish',  distillation=distillation, num_classes=num_classes)
    if pretrained:
        net.load_state_dict(torch.load(
            'weights/LeViT-128/checkpoint.pth')['model'])
    return net


@register_model
def LeViT_192(num_classes, distillation, pretrained=False):
    net = model_factory(C='192_288_384', D=32, N='3_5_6', X='4_4_4',
                        activation='Hardswish',  distillation=distillation, num_classes=num_classes)
    if pretrained:
        net.load_state_dict(torch.load(
            'weights/LeViT-192/checkpoint.pth')['model'])
    return net


@register_model
def LeViT_256(num_classes, distillation, pretrained=False):
    net = model_factory(C='256_384_512', D=32, N='4_6_8', X='4_4_4',
                        activation='Hardswish',  distillation=distillation, num_classes=num_classes)
    if pretrained:
        net.load_state_dict(torch.load(
            'weights/LeViT-256/checkpoint.pth')['model'])
    return net


@register_model
def LeViT_384(num_classes, distillation, pretrained=False):
    net = model_factory(C='384_576_768', D=32, N='6_9_12', X='4_4_4',
                        activation='Hardswish',  distillation=distillation, num_classes=num_classes)
    if pretrained:
        net.load_state_dict(torch.load(
            'weights/LeViT-384/checkpoint.pth')['model'])
    return net


FLOPS_COUNTER = 0


class Conv2d_BN(torch.nn.Module):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.c = torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(b)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

        global FLOPS_COUNTER
        output_points = ((resolution+2*pad-dilation*(ks-1)-1)//stride+1)**2
        FLOPS_COUNTER += a*b*output_points*(ks**2)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'cb'):
            del self.w
            del self.b
        else:
            w = self.bn.weight/(self.bn.running_var+self.bn.eps)**0.5
            b = self.bn.bias-self.bn.running_mean*self.bn.weight / \
                (self.bn.running_var+self.bn.eps)**0.5
            self.w = self.c.weight*w[:, None, None, None]
            self.b = b

    def forward(self, x):
        if self.training:
            return self.bn(self.c(x))
        else:
            return torch.nn.functional.conv2d(
                x, self.w,
                bias=self.b,
                stride=self.c.stride,
                padding=self.c.padding,
                dilation=self.c.dilation,
                groups=self.c.groups)


class Linear_BN(torch.nn.Module):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-100000):
        super().__init__()
        assert ks == 1
        assert stride == 1
        assert pad == 0
        assert dilation == 1
        assert groups == 1
        self.c = torch.nn.Linear(a, b, bias=False)
        self.bn = torch.nn.BatchNorm1d(b)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

        global FLOPS_COUNTER
        output_points = resolution**2
        FLOPS_COUNTER += a*b*output_points

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'cb'):
            del self.w
            del self.b
        else:
            w = self.bn.weight/(self.bn.running_var+self.bn.eps)**0.5
            b = self.bn.bias-self.bn.running_mean*self.bn.weight / \
                (self.bn.running_var+self.bn.eps)**0.5
            self.w = self.c.weight*w[:, None]
            self.b = b

    def forward(self, x):
        if self.training:
            x = self.c(x)
            return self.bn(x.flatten(0, 1)).reshape_as(x)
        else:
            return torch.nn.functional.linear(
                x, self.w, self.b)


class BN_Linear(torch.nn.Module):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(a)
        self.l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)

        global FLOPS_COUNTER
        FLOPS_COUNTER += a*b

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'w'):
            del self.w
            del self.b
        else:
            w = self.bn.weight/(self.bn.running_var+self.bn.eps)**0.5
            b = self.bn.bias-self.bn.running_mean*self.bn.weight / \
                (self.bn.running_var+self.bn.eps)**0.5
            self.w = self.l.weight*w[None, :]
            if self.l.bias is None:
                self.b = b@self.l.weight.T
            else:
                self.b = (self.l.weight@b[:, None]).view(-1)+self.l.bias

    def forward(self, x):
        if self.training:
            return self.l(self.bn(x))
        else:
            return torch.nn.functional.linear(
                x, self.w, self.b)


def b16(n, activation, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(3,   n//8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n//8, n//4, 3, 2, 1, resolution=resolution//2),
        activation(),
        Conv2d_BN(n//4, n//2, 3, 2, 1, resolution=resolution//4),
        activation(),
        Conv2d_BN(n//2, n,    3, 2, 1, resolution=resolution//8))


class Residual(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return x+self.m(x)


class Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim*num_heads
        self.d = int(attn_ratio*key_dim)
        self.dh = int(attn_ratio*key_dim)*num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd*2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * (resolution**4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**4)
        #attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution**4)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if True:
            if mode and hasattr(self, 'ab'):
                del self.ab
            else:
                self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Subsample(nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim*num_heads
        self.d = int(attn_ratio*key_dim)
        self.dh = int(attn_ratio*key_dim)*self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh+nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0]*stride-p2[0]+(size-1)/2),
                    abs(p1[1]*stride-p2[1]+(size-1)/2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**2) * (resolution_**2)
        #attention * v
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * self.d

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape

        k, v = self.kv(x).view(B, N, self.num_heads, -
                               1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)

        x = self.proj(x)
        return x


class LeViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2, 2, 2],
                 mlp_ratio=[2, 2, 2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.ReLU,
                 activation=torch.nn.ReLU,
                 **kwargs):
        super().__init__()
        global FLOPS_COUNTER

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim

        self.patch_embed = hybrid_backbone

        self.blocks = []
        down_ops.append([''])
        resolution = img_size//patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    )))
                if mr > 0:
                    h = int(ed*mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        )))
            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution-1)//do[5]+1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i+2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i+1]*do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i+1], h,
                                      resolution=resolution),
                            activation(),
                            Linear_BN(
                                h, embed_dim[i+1], bn_weight_init=0, resolution=resolution),
                        )))
        self.blocks = nn.Sequential(*self.blocks)

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.default_cfg = _cfg()

        print(FLOPS_COUNTER, 'flops')
        FLOPS_COUNTER = 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'conv_down_op' in x or 'attention_biases' in x}

    @torch.jit.export
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = x.mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LeViT_distillation(LeViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head_dist = BN_Linear(
            kwargs['embed_dim'][-1], kwargs['num_classes'])

    def forward(self, x):
        x = self.forward_features(x)
        x, xd = self.head(x), self.head_dist(x)
        if self.training:
            return x, xd
        else:
            return (x+xd)/2


def model_factory(C, D, X, N, activation, num_classes, distillation):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = getattr(torch.nn, activation)
    if distillation:
        l = LeViT_distillation
    else:
        l = LeViT
    model = l(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D]*3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0]//D, 2, 2, 2],
            ['Subsample', D, embed_dim[1]//D, 2, 2, 2],
        ],
        attention_activation=act,
        activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        num_classes=num_classes
    )
    return model


if __name__ == '__main__':
    net = LeViT_256(num_classes=1000, distillation=False).eval()
    print('Running LeViT-256 on a random tensor ...')
    net(torch.randn(1, 3, 224, 224))
    print('... done')
