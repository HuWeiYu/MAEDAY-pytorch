# Unofficial PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

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

Due to the limited gpus, it's really a chanllenge for us to pretrain with larger model or longer schedule mentioned in the paper. (the pretraining and end-to-end fine-tuning process of vit-large model are fininshed by [this enthusiastic handsome guy](https://github.com/sunsmarterjie) with many v100s, but the weights are unavailable)

So if one can fininsh it, please feel free to report it in the issue or push a PR, thank you!

And your star is my motivation, thank u~
