```
LeViT_128S 282662866 FLOPs 6944714 parameters
LeViT_128 383668800 FLOPs 8377112 parameters
LeViT_192 621541244 FLOPs 9961261 parameters
LeViT_256 1065926648 FLOPs 17326188 parameters
LeViT_384 2334044148 FLOPs 38567086 parameters


weights in /checkpoint/benjamingraham/LeViT/weights/*/model.pth

python main.py --eval --model LeViT_128S #* Acc@1 75.566 Acc@5 92.254 loss 1.006
python main.py --eval --model LeViT_128  #* Acc@1 77.420 Acc@5 93.392 loss 0.926
python main.py --eval --model LeViT_192  #* Acc@1 79.078 Acc@5 94.322 loss 0.845
python main.py --eval --model LeViT_256  #* Acc@1 81.068 Acc@5 95.284 loss 0.765
python main.py --eval --model LeViT_384  #* Acc@1 82.352 Acc@5 95.868 loss 0.727

Tesla V100-SXM2-16GB
timm.models.resnet50 cuda:0 2659.06704018743 images/s @ batch size 1024
timm.models.vit_deit_small_distilled_patch16_224 cuda:0 1904.8173045654205 images/s @ batch size 2048
timm.models.vit_deit_tiny_distilled_patch16_224 cuda:0 3924.9002281851103 images/s @ batch size 2048
levit.LeViT_128S cuda:0 13196.719755958686 images/s @ batch size 2048
levit.LeViT_128 cuda:0 9410.832706316309 images/s @ batch size 2048
levit.LeViT_192 cuda:0 8758.662004266567 images/s @ batch size 2048
levit.LeViT_256 cuda:0 6729.305903434312 images/s @ batch size 2048
levit.LeViT_384 cuda:0 4167.53046565594 images/s @ batch size 1024
timm.models.efficientnet_b0 cuda:0 4647.914214758268 images/s @ batch size 1024
timm.models.efficientnet_b1 cuda:0 2818.9689147239824 images/s @ batch size 1024
timm.models.efficientnet_b2 cuda:0 2106.107455818145 images/s @ batch size 512
timm.models.efficientnet_b3 cuda:0 1245.8019442634989 images/s @ batch size 512
timm.models.efficientnet_b4 cuda:0 593.3489423284612 images/s @ batch size 256

model name	: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
timm.models.resnet50 cpu 12.016911703481131 images/s @ batch size 16
timm.models.vit_deit_small_distilled_patch16_224 cpu 14.348798446856858 images/s @ batch size 16
timm.models.vit_deit_tiny_distilled_patch16_224 cpu 40.2349772036907 images/s @ batch size 16
levit.LeViT_128S cpu 144.1949651636412 images/s @ batch size 16
levit.LeViT_128 cpu 102.12269634884447 images/s @ batch size 16
levit.LeViT_192 cpu 77.26235422758357 images/s @ batch size 16
levit.LeViT_256 cpu 43.99343660045797 images/s @ batch size 16
levit.LeViT_384 cpu 24.25881057338392 images/s @ batch size 16
timm.models.efficientnet_b0 cpu 28.168883099150083 images/s @ batch size 16
timm.models.efficientnet_b1 cpu 19.573132817372812 images/s @ batch size 16
timm.models.efficientnet_b2 cpu 13.774425645372835 images/s @ batch size 16
timm.models.efficientnet_b3 cpu 6.052200595201753 images/s @ batch size 16
timm.models.efficientnet_b4 cpu 2.589793932115455 images/s @ batch size 16
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
