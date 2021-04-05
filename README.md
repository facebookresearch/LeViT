# LeViT

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


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


python main.py --eval --model LeViT_c_128S #* Acc@1 75.552 Acc@5 92.246 loss 1.006
python main.py --eval --model LeViT_c_128  #* Acc@1 77.400 Acc@5 93.388 loss 0.926
python main.py --eval --model LeViT_c_192  #* Acc@1 79.078 Acc@5 94.322 loss 0.845
python main.py --eval --model LeViT_c_256  #* Acc@1 81.066 Acc@5 95.292 loss 0.765
python main.py --eval --model LeViT_c_384  #* Acc@1 82.350 Acc@5 95.870 loss 0.727
```
