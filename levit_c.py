import torch
import itertools
import time
import os
import utils

from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model


def load_weights(net, path):
    d = torch.load(path)['model']
    D = net.state_dict()
    for k in d.keys():
        if D[k].shape != d[k].shape: 
            d[k] = d[k][:, :, None, None]
    net.load_state_dict(d)

@register_model
def LeViT_c_128S(num_classes, distillation=False, pretrained=False, fuse=False):
    net = model_factory(C='128_256_384', D=16, N='4_6_8', X='2_3_4',
                        activation='Hardswish', distillation=distillation, num_classes=num_classes)
    if pretrained:
        load_weights(net, '/checkpoint/benjamingraham/LeViT/weights/LeViT-128S-model.pth')
    if fuse:
        utils.replace_batchnorm(net)
    return net


@register_model
def LeViT_c_128(num_classes, distillation=False, pretrained=False, fuse=False):
    net = model_factory(C='128_256_384', D=16, N='4_8_12', X='4_4_4',
                        activation='Hardswish',  distillation=distillation, num_classes=num_classes)
    if pretrained:
        load_weights(net, '/checkpoint/benjamingraham/LeViT/weights/LeViT-128-model.pth')
    if fuse:
        utils.replace_batchnorm(net)
    return net


@register_model
def LeViT_c_192(num_classes, distillation=False, pretrained=False, fuse=False):
    net = model_factory(C='192_288_384', D=32, N='3_5_6', X='4_4_4',
                        activation='Hardswish',  distillation=distillation, num_classes=num_classes)
    if pretrained:
        load_weights(net, '/checkpoint/benjamingraham/LeViT/weights/LeViT-192-model.pth')
    if fuse:
        utils.replace_batchnorm(net)
    return net


@register_model
def LeViT_c_256(num_classes, distillation=False, pretrained=False, fuse=False):
    net = model_factory(C='256_384_512', D=32, N='4_6_8', X='4_4_4',
                        activation='Hardswish',  distillation=distillation, num_classes=num_classes)
    if pretrained:
        load_weights(net, '/checkpoint/benjamingraham/LeViT/weights/LeViT-256-model.pth')
    if fuse:
        utils.replace_batchnorm(net)
    return net


@register_model
def LeViT_c_384(num_classes, distillation=False, pretrained=False, fuse=False):
    net = model_factory(C='384_576_768', D=32, N='6_9_12', X='4_4_4',
                        activation='Hardswish',  distillation=distillation, num_classes=num_classes)
    if pretrained:
        load_weights(net, '/checkpoint/benjamingraham/LeViT/weights/LeViT-384-model.pth')
    if fuse:
        utils.replace_batchnorm(net)
    return net


FLOPS_COUNTER = 0


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution+2*pad-dilation*(ks-1)-1)//stride+1)**2
        FLOPS_COUNTER += a*b*output_points*(ks**2)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight/(bn.running_var+bn.eps)**0.5
        w = c.weight*w[:, None, None, None]
        b = bn.bias-bn.running_mean*bn.weight / (bn.running_var+bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a*b

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight/(bn.running_var+bn.eps)**0.5
        b = bn.bias-self.bn.running_mean * \
            self.bn.weight / (bn.running_var+bn.eps)**0.5
        w = l.weight*w[None, :]
        if l.bias is None:
            b = b@self.l.weight.T
        else:
            b = (l.weight@b[:, None]).view(-1)+self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


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


class Attention(torch.nn.Module):
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
        self.qkv = Conv2d_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(
            range(resolution), range(resolution)))
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
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x, attn=torch.zeros(0, dtype=torch.int)):  # x (B,C,H,W)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(
            B, self.num_heads, -1, H*W
        ).split([self.key_dim, self.key_dim, self.d], dim=2)
        attn = (
            (q.transpose(-2, -1)@k) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (v@attn.transpose(-2, -1)).view(B, -1, H, W)
        x = self.proj(x)
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
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
        h = self.d*num_heads+nh_kd
        self.kv = Conv2d_BN(in_dim, h, resolution=resolution)
        self.q = torch.nn.Sequential(
            torch.nn.AvgPool2d(1, stride, 0),
            Conv2d_BN(in_dim, nh_kd))
        self.proj = torch.nn.Sequential(
            activation(), Conv2d_BN(self.d*num_heads, out_dim, resolution=resolution_))

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

    def forward(self, x, attn=torch.zeros(0, dtype=torch.int)):
        B, C, H, W = x.shape
        k, v = self.kv(x).view(B, self.num_heads, -1, H *
                               W).split([self.key_dim, self.d], dim=2)
        q = self.q(x).view(B, self.num_heads, self.key_dim, self.resolution_2)

        attn = (q.transpose(-2, -1) @ k) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).reshape(
            B, -1, self.resolution_, self.resolution_)
        x = self.proj(x)
        return x


class LeViT(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 activation=torch.nn.Hardswish,
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
                            Conv2d_BN(ed, h, resolution=resolution),
                            activation(),
                            Conv2d_BN(h, ed, bn_weight_init=0,
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
                    h = int(embed_dim[i+1]*do[5])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Conv2d_BN(embed_dim[i+1], h,
                                      resolution=resolution),
                            activation(),
                            Conv2d_BN(
                                h, embed_dim[i+1], bn_weight_init=0, resolution=resolution),
                        )))
        self.blocks = torch.nn.Sequential(*self.blocks)

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
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
    for model in ['LeViT_c_128S', 'LeViT_c_128', 'LeViT_c_192', 'LeViT_c_256', 'LeViT_c_384']:
        net = globals()[model](num_classes=1000, distillation=False).eval()
        print(model, net.FLOPS, 'FLOPs', sum(p.numel()
                                             for p in net.parameters() if p.requires_grad), 'parameters')
        net(torch.randn(3,3,224,224))
