# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np
import torch


class RandomMaskingGenerator_N:
    def __init__(self, N, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.N = N
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask_torch_list = None
        for i in range(self.N):
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            mask_torch = torch.from_numpy(mask)
            if mask_torch_list is None:
                mask_torch_list = mask_torch.unsqueeze(0)
            else:
                mask_torch_list = torch.cat([mask_torch_list, mask_torch.unsqueeze(0)], dim=0)
        return mask_torch_list # [196]

if __name__ == '__main__':
    E = RandomMaskingGenerator_N(32,14,0.5)
    print(E.__call__().shape)