'''
Author: Jiyuan Liu
Date: 2025-02-08 17:59:27
LastEditors: WenJiawei
LastEditTime: 2025-03-05 17:10:37
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
        
class SquarePadAndResize(object):
    """将非正方形图片进行padding,使其成为正方形,再resize到指定尺寸。
    不足的部分补充黑边。
    """
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, x):
        if x.dim() == 3:
            C, H, W = x.shape
            max_side = max(H, W)
            pad_h = (max_side - H) // 2
            pad_w = (max_side - W) // 2
            # 计算需要补充的padding
            padding = [
                pad_w,  # 左边padding
                max_side - W - pad_w,  # 右边padding 
                pad_h,  # 上边padding
                max_side - H - pad_h,  # 下边padding
            ]
            # 进行padding,补充黑边(值为0)
            padded = F.pad(x, padding, mode='constant', value=0)
            padded = padded.unsqueeze(0)
            # resize到目标尺寸
            resized = F.interpolate(padded, size=self.target_size, mode='bicubic', align_corners=False)
            return resized.squeeze(0)
        elif x.dim() == 4:
            B, C, H, W = x.shape
            max_side = max(H, W)
            pad_h = (max_side - H) // 2
            pad_w = (max_side - W) // 2
            padding = [
                pad_w,
                max_side - W - pad_w,
                pad_h, 
                max_side - H - pad_h,
            ]
            padded = F.pad(x, padding, mode='constant', value=0)
            resized = F.interpolate(padded, size=self.target_size, mode='bicubic', align_corners=False)
            return resized
        else:
            raise ValueError("input tensor must be 3D or 4D tensor")
