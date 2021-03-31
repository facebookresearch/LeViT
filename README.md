# LeViT

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


```
weights in /checkpoint/benjamingraham/LeViT/weights/*/model.pth

python main.py --eval --model LeViT_128S #* Acc@1 75.566 Acc@5 92.254 loss 1.006 
python main.py --eval --model LeViT_128  #* Acc@1 77.420 Acc@5 93.392 loss 0.926 
python main.py --eval --model LeViT_192  #* Acc@1 79.078 Acc@5 94.322 loss 0.845 
python main.py --eval --model LeViT_256  #* Acc@1 81.068 Acc@5 95.284 loss 0.765 
python main.py --eval --model LeViT_384  #* Acc@1 82.352 Acc@5 95.868 loss 0.727
```
