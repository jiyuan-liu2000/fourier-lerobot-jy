'''
Author: Jiyuan Liu
Date: 2025-02-27 21:44:47
LastEditors: WenJiawei
LastEditTime: 2025-04-11 18:21:02
FilePath: /fourier-lerobot-jy/lerobot/common/policies/scaledp/modeling_scaledp.py
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
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


from huggingface_hub import PyTorchModelHubMixin

# from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.scaledp.configuration_scaledp import ScaleDPPolicyConfig
from lerobot.common.vision.dinov2 import DINOv2BackBone
from lerobot.common.policies.normalize import Normalize, Unnormalize, AdaptiveNormalize, AdaptiveUnnormalize
from lerobot.common.policies.utils import populate_queues, get_output_shape

import random
import os
import json
from pathlib import Path
from datetime import datetime

class ScaleDPPolicy(    
    nn.Module,  # 继承PyTorch的基础模块类
    PyTorchModelHubMixin,  # 继承HuggingFace模型仓库混入类
    library_name="lerobot",  # 指定库名称
    repo_url="https://github.com/huggingface/lerobot",  # 指定仓库URL
    tags=["robotics", "diffusion-policy"],  # 模型标签
    ):
    config_class = ScaleDPPolicyConfig
    name = "scale_dp"

    def __init__(
            self,
            config: ScaleDPPolicyConfig,
            dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        self.config = config
        

        # 1. 定义关节分组
        joint_groups = {
            "left_arm": [0, 1, 2, 3, 4, 5, 6],       # 左臂关节索引
            "right_arm": [7, 8, 9, 10, 11, 12, 13],  # 右臂关节索引
            "left_hand": [14, 15, 16, 17, 18, 19],   # 左手关节索引
            "right_hand": [20, 21, 22, 23, 24, 25]   # 右手关节索引
        }

        # 2. 定义缩放因子 - 手臂使用较小的缩放因子，手部使用较大的缩放因子
        scaling_factors = {
            "left_arm": 0.4,
            "right_arm": 0.4,
            "left_hand": 0.9,
            "right_hand": 0.9
        }

        # 3. 创建自适应归一化器
        self.normalize_inputs = AdaptiveNormalize(
            shapes=config.input_shapes, 
            modes=config.input_normalization_modes,
            stats=dataset_stats,
            joint_groups=joint_groups,
            scaling_factors=scaling_factors
        )

        self.normalize_targets = AdaptiveNormalize(
            shapes=config.output_shapes, 
            modes=config.output_normalization_modes,
            stats=dataset_stats,
            joint_groups=joint_groups,
            scaling_factors=scaling_factors
        )

        # 4. 创建自适应反归一化器
        self.unnormalize_outputs = AdaptiveUnnormalize(
            shapes=config.output_shapes,
            modes=config.output_normalization_modes,
            stats=dataset_stats,
            joint_groups=joint_groups,
            scaling_factors=scaling_factors
        )




        # self.normalize_inputs = Normalize(config.input_shapes, config.input_normalization_modes, dataset_stats)
        # self.normalize_targets = Normalize(
        #     config.output_shapes, config.output_normalization_modes, dataset_stats
        # )
        # self.unnormalize_outputs = Unnormalize(
        #     config.output_shapes, config.output_normalization_modes, dataset_stats
        # )


        ################################
        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        self.use_env_state = "observation.environment_state" in config.input_shapes

        self.model = ScaleDP(config)

        # 添加梯度跟踪相关属性
        self.grad_log_dir = Path("gradient_logs")
        self.grad_log_dir.mkdir(exist_ok=True)
        self.grad_stats = {
            "step": 0,
            "layers_with_small_grad": {},
            "layers_with_large_grad": {}
        }
        self.hooks = []

    def get_optim_params(self) -> dict:
        return self.model.get_optim_groups()
        # return self.model.parameters()
    
    def reset(self):
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if len(self.expected_image_keys) > 0:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.use_env_state:
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

    def register_gradient_hooks(self):
        """注册梯度钩子，用于监控特定层的梯度"""
        # 移除之前的钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # 跟踪关键组件的梯度
        tracked_modules = {
            "encoder": self.model.vision_encoder,
            "embedder": self.model.x_embedder,
            "t_embedder": self.model.t_embedder,
            "combine": self.model.combine,
            "blocks": self.model.blocks,
            "final_layer": self.model.final_layer
        }
        
        def grad_hook(name):
            def hook(grad):
                if grad is not None:
                    norm = grad.norm().item()
                    if norm < 1e-6:
                        if name not in self.grad_stats["layers_with_small_grad"]:
                            self.grad_stats["layers_with_small_grad"][name] = []
                        self.grad_stats["layers_with_small_grad"][name].append({
                            "step": self.grad_stats["step"],
                            "norm": norm
                        })
                    elif norm > 10.0:
                        if name not in self.grad_stats["layers_with_large_grad"]:
                            self.grad_stats["layers_with_large_grad"][name] = []
                        self.grad_stats["layers_with_large_grad"][name].append({
                            "step": self.grad_stats["step"],
                            "norm": norm
                        })
                return grad
            return hook
        
        # 为特定模块注册钩子
        for module_name, module in tracked_modules.items():
            if isinstance(module, nn.ModuleList):
                for i, block in enumerate(module):
                    for name, param in block.named_parameters():
                        if param.requires_grad:
                            self.hooks.append(param.register_hook(
                                grad_hook(f"{module_name}.{i}.{name}")
                            ))
            else:
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        self.hooks.append(param.register_hook(
                            grad_hook(f"{module_name}.{name}")
                        ))
        
        print(f"注册了 {len(self.hooks)} 个梯度钩子")
        
    def save_gradient_stats(self):
        """保存梯度统计信息到文件
        结果保存到路径 gradient_logs 下
        保存grad_stats_step_{step}_{timestamp}.json 梯度统计信息
        保存grad_summary_step_{step}_{timestamp}.txt 梯度问题摘要
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.grad_log_dir / f"grad_stats_step_{self.grad_stats['step']}_{timestamp}.json"
        
        # 限制记录数量，避免文件过大
        for key in ["layers_with_small_grad", "layers_with_large_grad"]:
            for layer_name in list(self.grad_stats[key].keys()):
                # 只保留最近的100条记录
                if len(self.grad_stats[key][layer_name]) > 100:
                    self.grad_stats[key][layer_name] = self.grad_stats[key][layer_name][-100:]
        
        with open(filename, 'w') as f:
            json.dump(self.grad_stats, f, indent=2)
        
        print(f"梯度统计信息已保存到 {filename}")
        
        # 创建梯度问题摘要
        summary_file = self.grad_log_dir / f"grad_summary_step_{self.grad_stats['step']}_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"=== 梯度问题摘要 (步骤 {self.grad_stats['step']}) ===\n\n")
            
            # 梯度过小的层
            f.write("梯度极小的层 (< 1e-6):\n")
            for layer_name, records in self.grad_stats["layers_with_small_grad"].items():
                if records:
                    avg_norm = sum(r["norm"] for r in records) / len(records)
                    f.write(f"  {layer_name}: 出现 {len(records)} 次, 平均范数: {avg_norm:.8f}\n")
            
            # 梯度过大的层
            f.write("\n梯度过大的层 (> 10.0):\n")
            for layer_name, records in self.grad_stats["layers_with_large_grad"].items():
                if records:
                    avg_norm = sum(r["norm"] for r in records) / len(records)
                    f.write(f"  {layer_name}: 出现 {len(records)} 次, 平均范数: {avg_norm:.2f}\n")
            
            # 问题模式分析
            f.write("\n问题模式分析:\n")
            small_grad_patterns = self._analyze_layer_patterns(self.grad_stats["layers_with_small_grad"])
            f.write("  梯度极小模式:\n")
            for pattern, count in small_grad_patterns.items():
                f.write(f"    {pattern}: {count} 层\n")
            
            large_grad_patterns = self._analyze_layer_patterns(self.grad_stats["layers_with_large_grad"])
            f.write("  梯度过大模式:\n")
            for pattern, count in large_grad_patterns.items():
                f.write(f"    {pattern}: {count} 层\n")
    
    def _analyze_layer_patterns(self, layer_dict):
        """分析层名称中的模式"""
        patterns = {}
        for layer_name in layer_dict.keys():
            # 提取模块类型
            parts = layer_name.split('.')
            if len(parts) >= 2:
                if parts[0] == "blocks":
                    pattern = f"transformer_block.{parts[2] if len(parts)>2 else 'general'}"
                else:
                    pattern = parts[0]
                
                patterns[pattern] = patterns.get(pattern, 0) + 1
        return patterns

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        import matplotlib.pyplot as plt
        import numpy as np

        # import pdb;pdb.set_trace()  
        
        ##################可视化 batch["observation.images"]
        # images = batch['observation.image.left'].cpu().numpy() # 维度：【batch_size, n_obs_steps, channels, height, width】
        # batch_size, n_obs_steps,channels, height, width  = images.shape

        # n_pic_show = min(n_obs_steps, 4)  # 最多显示4张图
        # fig, axes = plt.subplots(1, n_pic_show, figsize=(12, 3))  # 调整图像大小
        # for i in range(n_pic_show):
        #     ax = axes[i]
        #     # 调整维度顺序以正确显示图像
        #     img = images[0, i].transpose(1, 2, 0)  # 从(C,H,W)转换为(H,W,C)
        #     ax.imshow(img)
        #     ax.set_title(f'Step {i}')
        #     ax.axis('off')
        # plt.tight_layout()
        # plt.show()

        
        batch = self.normalize_inputs(batch)

        
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        
        batch = self.normalize_targets(batch)
        
        # 注册梯度钩子 (每100步注册一次，减少开销)
        if self.training and random.random() < 0.001:
            self.register_gradient_hooks()
        
        # 前向传播
        loss = self.model.compute_loss(batch)
        
        # 更新步骤计数
        if self.training:
            self.grad_stats["step"] += 1
            
            # 每500步保存一次梯度统计
            if self.grad_stats["step"] % 1000 == 0:
                self.save_gradient_stats()
                
            # 详细记录小梯度模块
            if random.random() < 0.01:  # 1%概率检查
                with torch.no_grad():
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                            grad_norm = module.weight.grad.norm().item()
                            if grad_norm < 1e-6:
                                print(f"模块 {name} 梯度极小: {grad_norm:.8f}")
                                
                                # 记录到列表中
                                if name not in self.grad_stats["layers_with_small_grad"]:
                                    self.grad_stats["layers_with_small_grad"][name] = []
                                
                                self.grad_stats["layers_with_small_grad"][name].append({
                                    "step": self.grad_stats["step"],
                                    "norm": grad_norm
                                })
        
        return {"loss": loss}

    def log_model_structure(self):
        """记录模型完整结构到日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        structure_file = self.grad_log_dir / f"model_structure_{timestamp}.txt"
        
        with open(structure_file, 'w') as f:
            f.write("=== ScaleDP模型结构 ===\n\n")
            
            # 主要组件
            f.write("主要组件:\n")
            f.write(f"  vision_encoder: {type(self.model.vision_encoder).__name__}\n")
            f.write(f"  x_embedder: {type(self.model.x_embedder).__name__}\n")  
            f.write(f"  t_embedder: {type(self.model.t_embedder).__name__}\n")
            f.write(f"  blocks: {len(self.model.blocks)}个Transformer块\n")
            f.write(f"  final_layer: {type(self.model.final_layer).__name__}\n\n")
            
            # 参数统计
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            f.write(f"总参数量: {total_params:,}\n")
            f.write(f"可训练参数量: {trainable_params:,}\n\n")
            
            # 详细层结构
            f.write("详细层结构:\n")
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.MultiheadAttention)):
                    params = sum(p.numel() for p in module.parameters())
                    f.write(f"  {name}: {type(module).__name__}, 参数量: {params:,}\n")
        
        print(f"模型结构已保存到 {structure_file}")


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
        identity = x
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        scale_factor = 0.2  # 残差缩放因子
        x = x + scale_factor * identity  # 额外的直接残差路径
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
        
        # import pdb; pdb.set_trace()
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        image_key = image_keys[0]
        dummy_input_h_w = (
            config.crop_shape if config.crop_shape is not None else config.input_shapes[image_key][1:]
        )
        dummy_input = torch.zeros(size=(1, config.input_shapes[image_key][0], *dummy_input_h_w))
        with torch.inference_mode():
            dummy_feature_map = self.backbone(dummy_input)
        if isinstance(dummy_feature_map, dict):
            self.use_feature_map_key = True
            dummy_feature_map = dummy_feature_map["feature_map"]
        else:
            self.use_feature_map_key = False
        feature_map_shape = tuple(dummy_feature_map.shape[1:])

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
        self.state_dim = self.config.input_shapes["observation.state"][0]
        self.action_dim = self.config.output_shapes["action"][0]
        
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

        # num_train_timesteps: int = 1000,
        # beta_start: float = 0.0001,
        # beta_end: float = 0.02,
        # beta_schedule: str = "linear",
        # trained_betas: ndarray | List[float] | None = None,
        # clip_sample: bool = True,
        # set_alpha_to_one: bool = True,
        # steps_offset: int = 0,
        # prediction_type: str = "epsilon",
        # thresholding: bool = False,
        # dynamic_thresholding_ratio: float = 0.995,
        # clip_sample_range: float = 1,
        # sample_max_value: float = 1,
        # timestep_spacing: str = "leading",
        # rescale_betas_zero_snr: bool = False

        if self.config.prediction_type == "epsilon":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=config.num_train_timesteps, # 100
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='epsilon',
                # prediction_type='v_prediction',
                timestep_spacing='trailing'
            )
        elif self.config.prediction_type == "v_prediction":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=config.num_train_timesteps, # 100
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                # prediction_type='epsilon',
                prediction_type='v_prediction',
                timestep_spacing='trailing')
        elif self.config.prediction_type == "sample":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=config.num_train_timesteps, # 100
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='sample',
                timestep_spacing='trailing')
        else :
            raise ValueError(f"不支持的预测类型: {self.config.prediction_type}。请使用 'epsilon'、'v_prediction' 或 'sample'。")
            
        # num_train_timesteps: int = 1000,
        # beta_start: float = 0.0001,
        # beta_end: float = 0.02,
        # beta_schedule: str = "linear",
        # trained_betas: ndarray | List[float] | None = None,
        # variance_type: str = "fixed_small",
        # clip_sample: bool = True,
        # prediction_type: str = "epsilon",
        # thresholding: bool = False,
        # dynamic_thresholding_ratio: float = 0.995,
        # clip_sample_range: float = 1,
        # sample_max_value: float = 1,
        # timestep_spacing: str = "leading",
        # steps_offset: int = 0,
        # rescale_betas_zero_snr: bool = False
        
        self.noise_scheduler_DDPM = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps, # 100
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            steps_offset=0,
            prediction_type='epsilon'
            # prediction_type='v_prediction'
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
        # import pdb; pdb.set_trace()
        batch_size, n_obs_steps = batch["observation.images"].shape[:2]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps, f"n_obs_steps:{n_obs_steps} != self.config.n_obs_steps:{self.config.n_obs_steps}"

        images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        
        # 计算图像数值范围
        # for i, images in enumerate(images_per_camera):
        #     min_val = torch.min(images).item()
        #     max_val = torch.max(images).item()
        #     mean_val = torch.mean(images).item()
        #     std_val = torch.std(images).item()
        #     print(f"相机 {i} 图像统计:")
        #     print(f"  最小值: {min_val:.3f}")
        #     print(f"  最大值: {max_val:.3f}") 
        #     print(f"  均值: {mean_val:.3f}")
        #     print(f"  标准差: {std_val:.3f}")
        
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

        if self.config.prediction_type == "epsilon" or  self.config.prediction_type == "sample":
            noisy_actions = torch.cat([self.noise_scheduler.add_noise(
                actions, noise[i], timesteps)
                for i in range(len(noise))], dim=0)  # [num_noise_samples * B, Ta, action_dim]
        elif self.config.prediction_type == "v_prediction":
            # import pdb; pdb.set_trace() 
            noisy_actions = torch.cat([self.noise_scheduler.add_noise(
                actions, noise[i], timesteps)
                for i in range(len(noise))], dim=0)  # [num_noise_samples * B, Ta, action_dim]
            velocity_gt = torch.cat([self.noise_scheduler.get_velocity(
                actions, noise[i], timesteps)
                for i in range(len(noise))], dim=0)  # [num_noise_samples * B, Ta, action_dim]

        noisy_actions = noisy_actions.to(dtype=actions.dtype)
        assert img_features.ndim == 3

        hidden_states = img_features.repeat(self.num_noise_samples, 1, 1) # [num_noise_samples * B, S, D]
        timesteps = timesteps.repeat(self.num_noise_samples)
        is_pad = batch["action_is_pad"].repeat(self.num_noise_samples, 1)
        # TODO: modify states to support multiple steps
        # import pdb; pdb.set_trace()
        # states = batch["observation.state"].squeeze().repeat(self.num_noise_samples, 1)
        states = batch["observation.state"].repeat(self.num_noise_samples, 1, 1)

        # import pdb;pdb.set_trace()
        noise_pred = self.model_forward(noisy_actions, timesteps, global_cond=hidden_states, states=states)
        noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])

        if self.config.prediction_type == "epsilon":
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
            loss = (loss * ~is_pad.unsqueeze(-1)).mean()
        elif self.config.prediction_type == "v_prediction":
            loss = torch.nn.functional.mse_loss(noise_pred, velocity_gt, reduction='none')
            loss = (loss * ~is_pad.unsqueeze(-1)).mean()
        elif self.config.prediction_type == "sample":
            # 计算原始样本和预测样本之间的MSE损失
            target_sample = actions.repeat(self.num_noise_samples, 1, 1)
            loss = torch.nn.functional.mse_loss(noise_pred, target_sample, reduction='none')
            loss = (loss * ~is_pad.unsqueeze(-1)).mean()
        return loss

    def generate_actions(self, batch: dict):
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")

        # ##################### 可视化 images_per_camera
        # import matplotlib.pyplot as plt
        # import numpy as np
        
        # # 获取第一个相机的图像
        # first_camera_images = images_per_camera[0].cpu().numpy()  # 维度: [(b*s), channels, height, width]
        
        # n_images = min(4, first_camera_images.shape[0])  # 最多显示4张图片
        
        # fig, axes = plt.subplots(1, n_images, figsize=(12, 3))
        # for i in range(n_images):
        #     ax = axes[i]
        #     # 调整维度顺序以正确显示图像
        #     img = first_camera_images[i].transpose(1, 2, 0)  # 从(C,H,W)转换为(H,W,C)
        #     ax.imshow(img)
        #     ax.set_title(f'Image {i}')
        #     ax.axis('off')
        # plt.tight_layout()
        # plt.show()

        # 计算图像统计信息
        first_camera_images = images_per_camera[0].cpu()  # 获取第一个相机的图像
        
        # 计算统计值
        mean_val = torch.mean(first_camera_images)
        min_val = torch.min(first_camera_images)
        max_val = torch.max(first_camera_images) 
        std_val = torch.std(first_camera_images)

        print(f"图像统计信息:")
        print(f"均值: {mean_val:.4f}")
        print(f"最小值: {min_val:.4f}")
        print(f"最大值: {max_val:.4f}")
        print(f"标准差: {std_val:.4f}")
        
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
        # import pdb; pdb.set_trace()

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
        # import pdb; pdb.set_trace()
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


        global_cond = global_cond[:,-1]
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

