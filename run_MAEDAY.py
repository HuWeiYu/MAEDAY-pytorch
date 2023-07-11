# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import tqdm
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import cv2
from PIL import Image

from timm.models import create_model

from MAEDAY.datasets import DataAugmentationForMAE
from MAEDAY.anoamly_detection import get_anomaly_map, calculate_Img_roc_auc

from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from MAEDAY import modeling_pretrain,modeling_finetune
def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--img_path', default='MVTec/bottle/test', type=str, help='input image path')
    parser.add_argument('--save_path', default='RES', type=str, help='save image path')
    parser.add_argument('--model_path', default='ckpt/pretrain/pretrain_mae_vit_base_mask_0.75_400e.pth', type=str, help='checkpoint path of model')
    parser.add_argument('--N', default=32, type=int,
                        help='MAE mask number')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


def main(args):
    print(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    transforms = DataAugmentationForMAE(args)

    gt_list, score_list = [],[]
    for each_anomaly_class in (os.listdir(args.img_path)):
        for each_test_pic in os.listdir(os.path.join(args.img_path,each_anomaly_class)):
            if each_anomaly_class == 'good':
                gt_list.append(0)
            else:
                gt_list.append(1)
            with open(os.path.join(args.img_path,each_anomaly_class,each_test_pic), 'rb') as f:
                img = Image.open(f)
                img.convert('RGB')

            img, bool_masked_pos = transforms(img)
            with torch.no_grad():
                query_img = img[None, :]
                query_img = query_img.to(device, non_blocking=True)
                # get original img numpy
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
                ori_img = query_img * std + mean  # in [0, 1]
                numpy_ori_img = ori_img[0, :].cpu().numpy().transpose(1,2,0)
                numpy_ori_img *= 255
                numpy_ori_img = numpy_ori_img.astype(np.uint8)
                numpy_ori_img = cv2.cvtColor(numpy_ori_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{args.save_path}/ori_img.jpg", numpy_ori_img)

                img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
                img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
                            img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')

                rec_img_list = []
                for each_MAE_MASK in bool_masked_pos:
                    each_MAE_MASK = each_MAE_MASK[None, :]
                    each_MAE_MASK = each_MAE_MASK.to(device, non_blocking=True).flatten(1).to(torch.bool)
                    outputs = model(query_img, each_MAE_MASK)
                    img_patch[each_MAE_MASK] = outputs

                    #get reconstruction img numpy
                    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
                    rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
                    rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)
                    numpy_rec_img = rec_img[0, :].clip(0,0.996).cpu().numpy()
                    numpy_rec_img = numpy_rec_img.transpose(1, 2, 0)
                    numpy_rec_img*=255
                    numpy_rec_img = numpy_rec_img.astype(np.uint8)
                    numpy_rec_img = cv2.cvtColor(numpy_rec_img, cv2.COLOR_RGB2BGR)
                    rec_img_list.append(numpy_rec_img)
                anomaly_map = get_anomaly_map(numpy_ori_img,rec_img_list)
                score_list.append(anomaly_map)
    image_level_roc_auc = calculate_Img_roc_auc(gt_list,score_list)
    print('image_level_roc_auc',image_level_roc_auc)


if __name__ == '__main__':
    opts = get_args()
    main(opts)
