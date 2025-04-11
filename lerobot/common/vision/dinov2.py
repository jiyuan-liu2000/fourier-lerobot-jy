'''
Author: Jiyuan Liu
Date: 2025-01-17 11:26:33
LastEditors: WenJiawei
LastEditTime: 2025-03-24 18:26:23
FilePath: /fourier-lerobot-jy/lerobot/common/vision/dinov2.py
Description: 

Copyright (c) 2024 by Fourier Intelligence Co. Ltd , All Rights Reserved. 
'''
import torch
from collections import OrderedDict
from torch import nn

class DINOv2BackBone(nn.Module):
    def __init__(self, return_dict=True) -> None:
        super().__init__()
        self.body = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.body.eval()
        self.num_channels = 384
        self.return_dict = return_dict 
    
    @torch.no_grad()
    def forward(self, tensor):
        # dinov2 patch size 16, the input tensor size should be divisible by 16
        xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
        # import pdb; pdb.set_trace() 
        od = OrderedDict()
        # od["0"] = xs.reshape(xs.shape[0], 16, 8, 384).permute(0, 3, 2, 1)
        od["0"] = xs.reshape(xs.shape[0], 16, 16, 384).permute(0, 3, 2, 1)
        # return od
        if self.return_dict:
            return {"feature_map": od["0"]}
        else:
            return od["0"]
    