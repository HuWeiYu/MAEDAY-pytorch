# Unofficial PyTorch implementation of [MAEDAY: MAE for few and zero shot AnomalY-Detection])

This repository is built upon [MAE](https://github.com/pengzhiliang/MAE-pytorch), thanks very much!
This repository currently only contain Zero-shot prediction.

## TODO
- [x] implement the lora training code
- [x] pixel-level roc_auc

## Setup
```
pip install -r requirements.txt
```

## Run
```
python run_MAEDAY.py
```
## Result
|   Catgory  | Paper(Img level) | The code(Img level) | 
|:--------:|:--------:|:--------:|
|  carpet  |          |          |
|  grid    |          |          |
|  leather |          |          |
|  tile    |          |          |
|  wood    |          |          |
|  bottle  |          |          |
|  cable   |          |          |
|  capsule |          |          |
|  hazelnut|          |          |
| metal nut|          |          |
|  pill    |          |          |
|  screw   |          |          |
| toothbrus|          |          |
|transistor|          |          |
|  zipper  |          |          |



## CKPT
|   model  | pretrain | finetune | accuracy | log | weight |
|:--------:|:--------:|:--------:|:--------:| :--------:|:--------:|
| vit-base |   400e   |   100e   |   83.1%  | [pretrain](files/pretrain_base_0.75_400e.txt) [finetune](files/pretrain_base_0.75_400e_finetune_100e.txt)| [Google drive](https://drive.google.com/drive/folders/182F5SLwJnGVngkzguTelja4PztYLTXfa?usp=sharing) [BaiduYun](https://pan.baidu.com/s/1F0u9WeckZMbNk095gUxT1g)(code: mae6)|


And your star is my motivation, thank u~
