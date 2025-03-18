'''
Author: Jiyuan Liu
Date: 2025-02-27 21:44:47
LastEditors: Jiyuan Liu
LastEditTime: 2025-03-17 14:00:57
FilePath: /fourier-lerobot/lerobot/common/policies/scaledp/modeling_scaledp.py
Description: 

Copyright (c) 2024 by Fourier Intelligence Co. Ltd , All Rights Reserved. 
'''
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import numpy as np
from collections import deque

import math
from typing import Tuple

import torch
import einops
import torchvision
from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from torch.jit import Final
from timm.models.vision_transformer import Mlp, use_fused_attn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.scaledp.configuration_scaledp import ScaleDPPolicyConfig
from lerobot.common.vision.dinov2 import DINOv2BackBone
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import populate_queues, get_output_shape

class ScaleDPPolicy(PreTrainedPolicy):
    config_class = ScaleDPPolicyConfig
    name = "scale_dp"

    def __init__(
            self,
            config: ScaleDPPolicyConfig,
            dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config = config
        
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = ScaleDP(config)

    def get_optim_params(self) -> dict:
        return self.model.get_optim_groups()
        # return self.model.parameters()
    
    def reset(self):
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch:dict[str, Tensor]) -> Tensor:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
                )
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.model.generate_actions(batch)

            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] if batch[key].dim == 5 else batch[key].unsqueeze(1) for key in self.config.image_features], dim=-4
                )
        batch = self.normalize_targets(batch)
        loss = self.model.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale

            attn_scores = torch.matmul(q, k.transpose(-2, -1))

            # Add attention mask if provided
            if attn_mask is not None:
                attn_scores += attn_mask

            # Apply softmax to get attention weights (softmax is applied along the last dimension)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Dropout on attention weights (if dropout is used)
            attn_weights = self.attn_drop(attn_weights)

            # Apply attention weights to value tensor (V)
            x = torch.matmul(attn_weights, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core ScaleDP Model                                #
#################################################################################

class ScaleDPBlock(nn.Module):
    """
    A ScaleDP block with adaptive layer norm zero (adaLN-Zero) conScaleDPioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask) # norm, scale&shift, attn, scale,
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of ScaleDP.
    """

    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class VisionEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: ScaleDPPolicyConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.resize_shape is not None:
            self.do_crop = True
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False
        self.img_transform = torchvision.transforms.Compose(
            [
                self.resize,
                self.maybe_random_crop,
            ]
        )

        # Set up backbone.
        if config.vision_backbone == "dino_v2":
            self.backbone = DINOv2BackBone(return_dict=False)
        else:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights
            ) 
            # Note: This assumes that the layer4 feature map is children()[-3]
            # TODO(alexander-soare): Use a safer alternative.
            self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        # self.feature_dim = config.cond_dim * len(self.config.image_features)
        self.feature_dim = config.cond_dim
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        # if self.do_crop:
        #     if self.training:  # noqa: SIM108
        #         x = self.maybe_random_crop(x)
        #     else:
        #         # Always use center crop for eval.
        #         x = self.center_crop(x)
        if self.do_crop:
            x = self.img_transform(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


class ScaleDP(nn.Module):
    """
    Diffusion models with a Transformer backbone.
    """
    def __init__(
            self,
            config: ScaleDPPolicyConfig,
    ):
        super().__init__()
        self.config = config
        
        # compute number of tokens for main trunk and conScaleDPion encoder
        if config.n_obs_steps is None:
            self.config.n_obs_steps = config.horizon
        T = config.horizon
        T_cond = 1
        if not config.time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = config.cond_dim > 0
        if obs_as_cond:
            assert config.time_as_cond
            T_cond += self.config.n_obs_steps

         # Backbone for image feature extraction.
        if self.config.image_features:

            # act format
            # num_images = len(self.config.image_features)
            # self.cond_dim = num_images * config.cond_dim
            # if config.vision_backbone == "dino_v2":
            #     self.backbone = DINOv2BackBone()
            #     self.encoder_img_feat_input_proj = nn.Conv2d(
            #         self.backbone.num_channels, self.cond_dim, kernel_size=1
            #     )
            # else:
            #     backbone_model = getattr(torchvision.models, config.vision_backbone)(
            #         replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            #         weights=config.pretrained_backbone_weights,
            #         norm_layer=FrozenBatchNorm2d,
            #     )
            #     # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            #     # feature map).
            #     # Note: The forward method of this returns a dict: {"feature_map": output}.
            #     self.backbone = IntermediateLayerGetter(
            #         backbone_model, return_layers={"layer4": "feature_map"}
            #     )
            #     self.encoder_img_feat_input_proj = nn.Conv2d(
            #         backbone_model.fc.in_features, self.cond_dim, kernel_size=1
            #     )

            # dp format
            num_images = len(self.config.image_features)
            encoders = [VisionEncoder(config) for _ in range(num_images)]
            self.vision_encoder = nn.ModuleList(encoders)
            self.cond_dim = encoders[0].feature_dim * num_images
        
        # get state_dim and action_dim
        if self.config.robot_state_feature:
            self.state_dim = self.config.robot_state_feature.shape[0]
        self.action_dim = self.config.action_feature.shape[0]
        
        self.is_tinyvla = config.is_tinyvla
        if config.is_tinyvla:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
            self.norm_after_pool = nn.LayerNorm(self.cond_dim)
        # self.combine = nn.Linear(cond_dim+state_dim, cond_dim)
        self.combine = nn.Sequential(
            nn.Linear(self.cond_dim+self.state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.cond_dim)
        )
        self.learn_sigma = config.learn_sigma
        self.input_dim = self.action_dim
        self.output_dim = self.action_dim * 2 if config.learn_sigma else self.action_dim
        self.num_heads = config.num_heads

        self.x_embedder = nn.Linear(self.action_dim, config.n_emb)
        self.t_embedder = TimestepEmbedder(config.n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(self.cond_dim, config.n_emb)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, config.horizon, config.n_emb))

        self.blocks = nn.ModuleList([
            ScaleDPBlock(config.n_emb, config.num_heads, mlp_ratio=config.mlp_ratio) for _ in range(config.depth)
        ])
        self.final_layer = FinalLayer(config.n_emb, output_dim=self.action_dim)
        self.initialize_weights()
        # constants
        self.T_cond = T_cond
        self.horizon = config.horizon
        self.time_as_cond = config.time_as_cond
        self.obs_as_cond = obs_as_cond
        # print(
        #     "number of parameters in ScaleDP: %e", sum(p.numel() for p in self.parameters())
        # )

        
        self.num_inference_timesteps = config.num_inference_timesteps
        # self.proj_to_action = nn.Identity()
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps, # 100
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        self.num_noise_samples = config.noise_samples # 1

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.cond_obs_emb.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.cond_obs_emb.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in ScaleDP blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the models into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, Attention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if "vision_encoder" in fpn:
                    if "bias" in pn or "bn" in mn or "bn" in fpn:
                        no_decay.add(fpn)
                    else:
                        decay.add(fpn)
                else:
                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.startswith("bias"):
                        # MultiheadAttention bias starts with "bias"
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # TODO: bug fix for pos_embed
        param_dict["pos_embed"] = self.pos_embed
        no_decay.add("pos_embed")
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-3,
                             betas: Tuple[float, float] = (0.9, 0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer
    
    def compute_loss(self, batch):
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch
        batch_size, n_obs_steps = batch["observation.images"].shape[:2]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps, f"n_obs_steps:{n_obs_steps} != self.config.n_obs_steps:{self.config.n_obs_steps}"

        images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.vision_encoder, images_per_camera, strict=True)
                    ]
                )
        img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                ) #(B, S, D)
        
        actions = batch["action"] 
        states = batch["observation.state"]
        noise = torch.randn([self.num_noise_samples] + list(actions.shape), device=actions.device,
                                dtype=actions.dtype)
        
        timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=actions.device
            ).long()
        
        timesteps, noise = timesteps.to(actions.device), noise.to(actions.device)
        noisy_actions = torch.cat([self.noise_scheduler.add_noise(
            actions, noise[i], timesteps)
            for i in range(len(noise))], dim=0)  # [num_noise_samples * B, Ta, action_dim]

        noisy_actions = noisy_actions.to(dtype=actions.dtype)
        assert img_features.ndim == 3

        hidden_states = img_features.repeat(self.num_noise_samples, 1, 1) # [num_noise_samples * B, S, D]
        timesteps = timesteps.repeat(self.num_noise_samples)
        is_pad = batch["action_is_pad"].repeat(self.num_noise_samples, 1)
        # TODO: modify states to support multiple steps
        states = batch["observation.state"].squeeze().repeat(self.num_noise_samples, 1)

        noise_pred = self.model_forward(noisy_actions, timesteps, global_cond=hidden_states, states=states)
        noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
        loss = (loss * ~is_pad.unsqueeze(-1)).mean()
        return loss

    def generate_actions(self, batch: dict):
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.vision_encoder, images_per_camera, strict=True)
                    ]
                )
        img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                ) #(B, S, D)
        states = batch["observation.state"]
        
        # initialize action from Guassian noise
        noisy_action = torch.randn((batch_size, self.horizon, self.action_dim)).cuda()

        naction = noisy_action.to(dtype=img_features.dtype)
        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.model_forward(naction, k, global_cond=img_features, states=states)

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        return naction

    def model_forward(self, x, t, global_cond, states):
        """
        Forward pass of ScaleDP.
        x: (N, T, input_dim) noisy actions
        t: (N,) tensor of diffusion timesteps
        global_cond: (N, n_obs_steps, D) tensor of conScaleDPions: image embeddings
        """
        if self.is_tinyvla:
            global_cond = self.global_1d_pool(global_cond.permute(0, 2, 1)).squeeze(-1)
            global_cond = self.norm_after_pool(global_cond)
        else:
            global_cond = global_cond.squeeze(1)
        global_cond = torch.cat([global_cond, states], dim=-1) if states is not None else global_cond
        global_cond = self.combine(global_cond)

        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        elif torch.is_tensor(t) and len(t.shape) == 0:
            t = t[None].to(x.device)
        t = t.expand(t.shape[0])

        x = self.x_embedder(x) + self.pos_embed.to(device=x.device, dtype=x.dtype)  # (N, T, D), where T = prediction_horizon
        t = self.t_embedder(t)  # (N, D)
        if self.obs_as_cond:
            global_cond = self.cond_obs_emb(global_cond)  # (N, D)
        # c = t + global_cond.sum(dim=1)  # (N, D)
        c = t + global_cond  # (N, D)
        for block in self.blocks:
            # x = block(x, c, attn_mask=self.mask)  # (N, T, D)
            x = block(x, c, attn_mask=None)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, output_dim)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

from typing import Callable
def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

