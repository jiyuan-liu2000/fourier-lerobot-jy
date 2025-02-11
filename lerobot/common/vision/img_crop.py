'''
Author: Jiyuan Liu
Date: 2025-02-08 17:59:27
LastEditors: Jiyuan Liu
LastEditTime: 2025-02-08 17:59:29
FilePath: /fourier-lerobot-jy/lerobot/common/vision/img_crop.py
Description: 

Copyright (c) 2024 by Fourier Intelligence Co. Ltd , All Rights Reserved. 
'''
import torch
import torch.nn.functional as F

class SquareCenterCropAndResize(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, x):
        if x.dim() == 3:
            C, H, W = x.shape
            side = min(H, W)
            top = (H - side) // 2
            left = (W - side) // 2
            cropped = x[:, top:top+side, left:left+side]
            cropped = cropped.unsqueeze(0)
            resized = F.interpolate(cropped, size=self.target_size, mode='bicubic', align_corners=False)
            return resized.squeeze(0) 
        elif x.dim() == 4:
            B, C, H, W = x.shape
            side = min(H, W)
            top = (H - side) // 2
            left = (W - side) // 2
            cropped = x[:, :, top:top+side, left:left+side]
            resized = F.interpolate(cropped, size=self.target_size, mode='bicubic', align_corners=False)
            return resized 
        else:
            raise ValueError("input tensor must be 3D or 4D tensor")