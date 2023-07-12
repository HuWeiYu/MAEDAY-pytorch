# Unofficial PyTorch implementation of [MAEDAY: MAE for few and zero shot AnomalY-Detection])

This repository is built upon [MAE](https://github.com/pengzhiliang/MAE-pytorch), thanks very much!
This repository currently only contain Zero-shot prediction.

## TODO
- [x] implement the lora training code
- [x] pixel-level auroc

## Setup
```
pip install -r requirements.txt
```

## Run
```
python run_MAEDAY.py
```
## Result
|   Catgory  | Paper(Img level) | This code(Img level) | 
|:--------:|:--------:|:--------:|
|  carpet  |   74.6       |    45.5       |
|  grid    |    97.9        |   68.6      |
|  leather |    92.9      |    61.5      |
|  tile    |     84.3     |     51.8     |
|  wood    |     94.8     |   56.5       |
|  bottle  |  74.3        |    87.4      |
|  cable   |    53.0      |    39.6      |
|  capsule |   64.0       |    51.4     |
|  hazelnut|      97.1    |    89.3      |
| metal_nut|       43.6   |     46.7     |
|  pill    |     63.4       |   56.9      |
|  screw   |     69.9     |   52.8       |
| toothbrush|     77.5     |   42.8       |
|transistor|      48.3    |    49.2      |
|  zipper  |       82.0   |     64.3     |

As you can see, there is quite a margin between us, maybe that's because there is some tricks that the origin auther that don't mention in the paper. There is a （+-5）in the Img-level roc_auc in this code， because the 32 MAE mask is generated randomly， that some mask maybe perfect (it covers the overall defect area), so the score will rise, if you are lucky. I don't run like 10 time to get a mean value.


## CKPT
|   model  | pretrain | finetune | accuracy | log | weight |
|:--------:|:--------:|:--------:|:--------:| :--------:|:--------:|
| vit-base |   400e   |   100e   |   83.1%  | [pretrain](files/pretrain_base_0.75_400e.txt) [finetune](files/pretrain_base_0.75_400e_finetune_100e.txt)| [Google drive](https://drive.google.com/drive/folders/182F5SLwJnGVngkzguTelja4PztYLTXfa?usp=sharing) [BaiduYun](https://pan.baidu.com/s/1F0u9WeckZMbNk095gUxT1g)(code: mae6)|


And your star is my motivation, thank u~
