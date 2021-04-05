```
LeViT_128S 282662866 FLOPs 6963770 parameters
LeViT_128 383668800 FLOPs 8400648 parameters
LeViT_192 621541244 FLOPs 9987989 parameters
LeViT_256 1065926648 FLOPs 17361484 parameters
LeViT_384 2334044148 FLOPs 38620030 parameters


weights in /checkpoint/benjamingraham/LeViT/weights/*/model.pth

python main.py --eval --model LeViT_128S #* Acc@1 75.566 Acc@5 92.254 loss 1.006 
python main.py --eval --model LeViT_128  #* Acc@1 77.420 Acc@5 93.392 loss 0.926 
python main.py --eval --model LeViT_192  #* Acc@1 79.078 Acc@5 94.322 loss 0.845 
python main.py --eval --model LeViT_256  #* Acc@1 81.068 Acc@5 95.284 loss 0.765 
python main.py --eval --model LeViT_384  #* Acc@1 82.352 Acc@5 95.868 loss 0.727


model name	: Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
timm.models.resnet50 cpu 7.944002064457476 images/s @ batch size 16
timm.models.vit_deit_tiny_distilled_patch16_224 cpu 28.001897284790186 images/s @ batch size 16
timm.models.vit_deit_small_distilled_patch16_224 cpu 8.711776239996817 images/s @ batch size 16
levit.LeViT_128S cpu 106.10848851642947 images/s @ batch size 16
levit.LeViT_128 cpu 76.13930432645435 images/s @ batch size 16
levit.LeViT_192 cpu 55.23201346462578 images/s @ batch size 16
levit.LeViT_256 cpu 31.310483975850254 images/s @ batch size 16
levit.LeViT_384 cpu 15.801694443129772 images/s @ batch size 16
timm.models.efficientnet_b0 cpu 22.682007389944054 images/s @ batch size 16
timm.models.efficientnet_b1 cpu 16.72780311508573 images/s @ batch size 16
timm.models.efficientnet_b2 cpu 12.06568945093711 images/s @ batch size 16
timm.models.efficientnet_b3 cpu 5.458859787492126 images/s @ batch size 16
timm.models.efficientnet_b4 cpu 2.2449595773316195 images/s @ batch size 16


Tesla V100-SXM2-16GB
timm.models.resnet50 cuda:0 2696.72981096612 images/s @ batch size 1024
timm.models.vit_deit_tiny_distilled_patch16_224 cuda:0 3966.037528742737 images/s @ batch size 2048
timm.models.vit_deit_small_distilled_patch16_224 cuda:0 1920.3454305211385 images/s @ batch size 2048
levit.LeViT_128S cuda:0 13368.443936099051 images/s @ batch size 2048
levit.LeViT_128 cuda:0 9526.683232112708 images/s @ batch size 2048
levit.LeViT_192 cuda:0 8913.585777378386 images/s @ batch size 2048
levit.LeViT_192B cuda:0 8175.149964465427 images/s @ batch size 2048
levit.LeViT_256 cuda:0 6838.016070256175 images/s @ batch size 2048
levit.LeViT_256B cuda:0 6129.510691136135 images/s @ batch size 2048
levit.LeViT_384 cuda:0 4235.291509223601 images/s @ batch size 1024
timm.models.efficientnet_b0 cuda:0 4716.660634850059 images/s @ batch size 1024
timm.models.efficientnet_b1 cuda:0 2858.016103163522 images/s @ batch size 1024
timm.models.efficientnet_b2 cuda:0 2130.9554249849602 images/s @ batch size 512
timm.models.efficientnet_b3 cuda:0 1261.8188400682386 images/s @ batch size 512
timm.models.efficientnet_b4 cuda:0 601.735645220687 images/s @ batch size 256
```


# LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference

This repository contains PyTorch evaluation code, training code and pretrained models for LeViT.

They obtain competitive tradeoffs in terms of speed / precision:

![LeViT](.github/levit.png)

For details see [LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference](https://arxiv.org/abs/2104.00000) by Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou and Matthijs Douze.

If you use this code for a paper please cite:

```
@article{graham2021levit,
  title={LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference},
  author={Benjamin Graham and Alaaeldin El-Nouby and Hugo Touvron and Pierre Stock and Armand Joulin and Herv\'e J\'egou and Matthijs Douze},
  journal={arXiv preprint arXiv:2021.00000},
  year={2021}
}
```

# Model Zoo

We provide baseline LeViT  models trained with distllation on ImageNet 2012. 

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| LeViT-128S | 75.6 |  92.3 | 7.0M | [model](https://dl.fbaipublicfiles.com/LeViT/) |
| LeViT-128  | 77.4 |  93.4 | 8.4M | [model](https://dl.fbaipublicfiles.com/LeViT/) |
| LeViT-192  | 79.1 |  94.3 | 10M | [model](https://dl.fbaipublicfiles.com/LeViT/) |
| LeViT-256  | 81.1 |  95.3 | 17M | [model](https://dl.fbaipublicfiles.com/LeViT/) |
| LeViT-384  | 82.4 |  95.9 | 39M | [model](https://dl.fbaipublicfiles.com/LeViT/) |


# Usage

First, clone the repository locally:
```
git clone https://github.com/facebookresearch/levit.git
```
Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Evaluation
To evaluate a pre-trained LeViT-256 model on ImageNet val with a single GPU run:
```
python main.py --eval --resume https://dl.fbaipublicfiles.com/levit/.... --data-path /path/to/imagenet
```
This should give
```
* Acc@1 81.164 Acc@5 95.376 loss 0.752
```


## Training
To train LeViT-256 on ImageNet with hard distillation on a single node with 8 gpus for 500 epochs run:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --data-path /path/to/imagenet --output_dir /path/to/save
```

### Multinode training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train LeViT-256 model on ImageNet on one nodes with 8 gpus for 500 epochs:

```
python run_with_submitit.py --data-path /path/to/imagenet
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
