'''
Author: Jiyuan Liu
Date: 2025-02-24 15:21:53
LastEditors: Jiyuan Liu
LastEditTime: 2025-02-25 16:58:46
FilePath: /fourier-lerobot-jy/lerobot/common/policies/scaledp/configuration_scaledp.py
Description: 

Copyright (c) 2024 by Fourier Intelligence Co. Ltd , All Rights Reserved. 
'''
import os
from typing import Union, List
from dataclasses import dataclass, field

from transformers.utils import logging
from transformers import AutoConfig, AutoModelForCausalLM
logger = logging.get_logger(__name__)

MODEL_STRUCTURE = {
    'ScaleDP_H': {'depth': 32, 'n_emb': 1280, 'num_heads': 16, },
    'ScaleDP_L': {'depth': 24, 'n_emb': 1024, 'num_heads': 16, }, # 400M
}

@dataclass
class ScaleDPPolicyConfig():
    '''
    Configuration for ScaleDP policy head
    '''
    n_obs_steps: int = 2
    prediction_horizon: int = 16
    n_action_steps: int = 8

    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.image": [3, 224, 224],
            "observation.state": [26],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [26],
        }
    )

    eval:bool = False

    state_dim:int = input_shapes["observation.state"]
    input_dim:int = output_shapes["action"]
    cond_dim:int = 1563 # the input dim of the condition
    output_dim:int = output_shapes["action"]

    depth:int = 32
    n_emb: int = 1280
    num_heads: int = 16

    model_size: str = "ScaleDP_H"
    if model_size != "none":
            depth = MODEL_STRUCTURE[model_size]['depth']
            n_emb = MODEL_STRUCTURE[model_size]['n_emb']
            num_heads = MODEL_STRUCTURE[model_size]['num_heads']
    else:
        raise ValueError("model_size show not be 'none'")
    
    mlp_ratio: int = 4.0,
    time_as_cond: bool = True
    obs_as_cond: bool = True
    learn_sigma: bool = False
    num_inference_timesteps: int = 10
    noise_samples: int = 1
    num_train_timesteps: int = 100
    is_tinyvla:bool = False
    num_queries:int = prediction_horizon

    def __post_init__(self):
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        
        image_keys = {k for k in self.input_shapes if k.startswith("observation.image")}

        if len(image_keys) == 0 and "observation.environment_state" not in self.input_shapes:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if len(image_keys) > 0:
            if self.crop_shape is not None:
                for image_key in image_keys:
                    if (
                        self.crop_shape[0] > self.input_shapes[image_key][1]
                        or self.crop_shape[1] > self.input_shapes[image_key][2]
                    ):
                        import warnings
                        # raise ValueError(
                        warnings.warn(
                            f"`crop_shape` should fit within `input_shapes[{image_key}]`. Got {self.crop_shape} "
                            f"for `crop_shape` and {self.input_shapes[image_key]} for "
                            f"`input_shapes[{image_key}]`."
                        )
            # Check that all input images have the same shape.
            first_image_key = next(iter(image_keys))
            for image_key in image_keys:
                if self.input_shapes[image_key] != self.input_shapes[first_image_key]:
                    raise ValueError(
                        f"`input_shapes[{image_key}]` does not match `input_shapes[{first_image_key}]`, but we "
                        "expect all image shapes to match."
                    )


# class ScaleDPPolicyConfig():
#     '''
#     Configuration for ScaleDP policy head
#     '''
#     model_type = "scale_dp_policy"
#     def __init__(
#             self,
#             eval: bool = False,
#             action_dim: int = 14,  # action dim
#             cond_dim: int = 1536,  # the input dim of the condition
#             state_dim: int = 14,  # the input dim of the state
#             horizon: int = 16,  # horizon
#             n_obs_steps: int = 2,  # number of observation steps
#             depth: int = 32,  # number of DiT blocks
#             n_emb: int = 1280,  # embedding size
#             num_heads: int = 16, 
#             mlp_ratio: int = 4.0,
#             time_as_cond: bool = True,
#             obs_as_cond: bool = True,
#             learn_sigma: bool = False,
#             model_size: str = "ScaleDP_H",
#             num_inference_timesteps: int = 10,
#             noise_samples: int = 1,
#             num_train_timesteps: int = 100,
#             is_tinyvla: bool = False,
#             **kwargs
#     ):
#         if model_size != "none":
#             depth = MODEL_STRUCTURE[model_size]['depth']
#             n_emb = MODEL_STRUCTURE[model_size]['n_emb']
#             num_heads = MODEL_STRUCTURE[model_size]['num_heads']
#         else:
#             raise ValueError("model_size show not be 'none'")
#         self.eval = eval

#         self.input_dim = action_dim
#         self.output_dim = action_dim
#         self.prediction_horizon = horizon

#         self.is_tinyvla = is_tinyvla

#         self.cond_dim = cond_dim
#         self.state_dim = state_dim

#         self.n_obs_steps = n_obs_steps
#         self.depth = depth
#         self.n_emb = n_emb
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         self.time_as_cond = time_as_cond
#         self.obs_as_cond = obs_as_cond
#         self.learn_sigma = learn_sigma

#         self.num_inference_timesteps = num_inference_timesteps
#         self.num_queries = horizon
#         self.noise_samples = noise_samples
#         self.num_train_timesteps = num_train_timesteps
#         super().__init__(**kwargs)

#     def __post_init__(self):
#         if not self.vision_backbone.startswith("resnet"):
#             raise ValueError(
#                 f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
#             )
        
#         image_keys = {k for k in self.input_shapes if k.startswith("observation.image")}

#         if len(image_keys) == 0 and "observation.environment_state" not in self.input_shapes:
#             raise ValueError("You must provide at least one image or the environment state among the inputs.")

#         if len(image_keys) > 0:
#             if self.crop_shape is not None:
#                 for image_key in image_keys:
#                     if (
#                         self.crop_shape[0] > self.input_shapes[image_key][1]
#                         or self.crop_shape[1] > self.input_shapes[image_key][2]
#                     ):
#                         import warnings
#                         # raise ValueError(
#                         warnings.warn(
#                             f"`crop_shape` should fit within `input_shapes[{image_key}]`. Got {self.crop_shape} "
#                             f"for `crop_shape` and {self.input_shapes[image_key]} for "
#                             f"`input_shapes[{image_key}]`."
#                         )