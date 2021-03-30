# srun --gres=gpu:8 --partition dev --constraint=volta16gb -c80 -t100 --mem 500GB --pty python speed_test.py | tee timings
import os
import apex
import torch
import torchvision
import time
import timm
import models_BN3000L
import torchvision
torch.autograd.set_grad_enabled(False)
print(torch.cuda.get_device_properties(0).total_memory)

os.system('echo -n "nb processors "; '
          'cat /proc/cpuinfo | grep ^processor | wc -l; '
          'cat /proc/cpuinfo | grep ^"model name" | tail -1')


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Sequential())
        else:
            replace_batchnorm(child)


def replace_layernorm(net):
    for child_name, child in net.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            setattr(net, child_name, apex.normalization.FusedLayerNorm(
                child.weight.size(0)))
        else:
            replace_layernorm(child)


T0 = 1
T1 = 3


def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    start = time.time()
    while time.time()-start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)


def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time()-start < T0:
            model(inputs)
    timing = []
    if device == 'cuda:0':
        torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)


for threads in [-1, 1]:
    if threads > 0:
        print(threads, 'threads')
        torch.set_num_threads(threads)
        device = 'cpu'
        compute_throughput = compute_throughput_cpu
    else:
        print('cuda')
        device = 'cuda:0'
        compute_throughput = compute_throughput_cuda

    for n, batch_size0, kwargs, resolution in [
        ('resnet50', 1024, {}, 224),
        #('LeViT', 2048, {'C': '128_192_256', 'D': 16, 'N': '4_6_8', 'X': '4_4_4', 'act': 'Hardswish', 'num_classes': 1000}, 224),
        ('LeViT', 2048, {'C': '128_256_384', 'D': 16, 'N': '4_6_8',
                         'X': '2_3_4', 'act': 'Hardswish', 'num_classes': 1000}, 224),
        ('LeViT', 2048, {'C': '128_256_384', 'D': 16, 'N': '4_8_12',
                         'X': '4_4_4', 'act': 'Hardswish', 'num_classes': 1000}, 224),
        ('LeViT', 2048, {'C': '192_288_384', 'D': 32, 'N': '3_5_6',
                         'X': '4_4_4', 'act': 'Hardswish', 'num_classes': 1000}, 224),
        ('LeViT', 2048, {'C': '256_384_512', 'D': 32, 'N': '4_6_8',
                         'X': '4_4_4', 'act': 'Hardswish', 'num_classes': 1000}, 224),
        ('LeViT', 1024, {'C': '384_576_768', 'D': 32, 'N': '6_9_12',
                         'X': '4_4_4', 'act': 'Hardswish', 'num_classes': 1000}, 224),
        ('vit_deit_tiny_distilled_patch16_224', 2048, {}, 224),
        ('vit_deit_small_distilled_patch16_224', 2048, {}, 224),
        ('efficientnet_b0',   1024, {}, 224),
        ('efficientnet_b1',   1024, {}, 240),
        ('efficientnet_b2',   512, {}, 260),
        ('efficientnet_b3',   512, {}, 300),
        ('efficientnet_b4',   256, {}, 380),
    ]:
        if device == 'cpu':
            batch_size = 16
        else:
            batch_size = batch_size0
            torch.cuda.empty_cache()
        inputs = torch.randn(batch_size, 3, resolution,
                             resolution, device=device)
        for m in [models_BN3000L, timm.models]:
            if hasattr(m, n):
                model = getattr(m, n)(**kwargs)
        if 'efficientnet_' in n:
            replace_batchnorm(model)
        if device != 'cpu' and 'deit' in n:
            replace_layernorm(model)
        model.to(device)
        model.eval()
        model = torch.jit.trace(model, inputs)
        compute_throughput(n+str(kwargs), model, device,
                           batch_size, resolution=resolution)
        print()
