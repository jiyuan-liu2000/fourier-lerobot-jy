import torch
from collections import OrderedDict
from torch import nn

class DINOv2BackBone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.body.eval()
        self.num_channels = 384
    
    @torch.no_grad()
    def forward(self, tensor):
        xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
        od = OrderedDict()
        od["0"] = xs.reshape(xs.shape[0], 22, 16, 384).permute(0, 3, 2, 1)
        # return od
        return {"feature_map": od["0"]}
    