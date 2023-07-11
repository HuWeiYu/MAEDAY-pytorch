# Unofficial PyTorch implementation of [MAEDAY: MAE for few and zero shot AnomalY-Detection])

This repository is built upon [MAE](https://github.com/pengzhiliang/MAE-pytorch), thanks very much!
This repository currently only contain Zero-shot prediction.

## TODO
- [x] implement the lora training code
- [ ] ...

## Setup
```
pip install -r requirements.txt
```

## Run
```
python run_MAEDAY.py
```

## CKPT
|   model  | pretrain | finetune | accuracy | log | weight |
|:--------:|:--------:|:--------:|:--------:| :--------:|:--------:|
| vit-base |   400e   |   100e   |   83.1%  | [pretrain](files/pretrain_base_0.75_400e.txt) [finetune](files/pretrain_base_0.75_400e_finetune_100e.txt)| [Google drive](https://drive.google.com/drive/folders/182F5SLwJnGVngkzguTelja4PztYLTXfa?usp=sharing) [BaiduYun](https://pan.baidu.com/s/1F0u9WeckZMbNk095gUxT1g)(code: mae6)|
| vit-large | 400e | 50e | 84.5% | [pretrain](files/pretrain_large_0.75_400e.txt) [finetune](files/pretrain_large_0.75_400e_finetune_50e.txt) | unavailable |

And your star is my motivation, thank u~
