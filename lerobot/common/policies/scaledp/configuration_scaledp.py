'''
Author: Jiyuan Liu
Date: 2025-02-27 21:44:47
LastEditors: WenJiawei
LastEditTime: 2025-03-19 17:49:32
FilePath: /fourier-lerobot-jy/lerobot/common/policies/scaledp/configuration_scaledp.py
Description: 

Copyright (c) 2024 by Fourier Intelligence Co. Ltd , All Rights Reserved. 
'''
from dataclasses import dataclass, field

# from lerobot.common.optim.optimizers import AdamConfig
# from lerobot.common.optim.schedulers import DiffuserSchedulerConfig
# from lerobot.configs.policies import PreTrainedConfig
# from lerobot.configs.types import NormalizationMode

MODEL_STRUCTURE = {
    'ScaleDP_Ti': {'depth': 8, 'n_emb': 256, 'num_heads': 4, }, # 10M
    'ScaleDP_S': {'depth': 12, 'n_emb': 384, 'num_heads': 6, }, # 33M
    'ScaleDP_B': {'depth': 12, 'n_emb': 768, 'num_heads': 12, }, # 130M
    'ScaleDP_L': {'depth': 24, 'n_emb': 1024, 'num_heads': 16, }, # 457M
    'ScaleDP_H': {'depth': 32, 'n_emb': 1280, 'num_heads': 16, }, # 1B
}
# @PreTrainedConfig.register_subclass("scale_dp")
@dataclass
class ScaleDPPolicyConfig():
    '''
    Configuration for ScaleDP policy head
    '''
    n_obs_steps: int = 3  # number of observation steps
    horizon: int = 32
    n_action_steps: int = 8

    model_size: str = "none"
    
    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.image": [3, 96, 96],
            "observation.state": [2],
        }
    )
    image_features: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.image": [3, 96, 96],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [2],
        }
    )
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.image": "mean_std",
            "observation.state": "min_max",
        }
    )
    output_normalization_modes: dict[str, str] = field(default_factory=lambda: {"action": "min_max"})

    # normalization_mapping: dict[str, NormalizationMode] = field(
    #     default_factory=lambda: {
    #         "VISUAL": NormalizationMode.MEAN_STD,
    #         "STATE": NormalizationMode.MEAN_STD,
    #         "ACTION": NormalizationMode.MEAN_STD,
    #     }
    # )

    # vision backbone
    vision_backbone: str = "resnet50"
    replace_final_stride_with_dilation: bool = False
    pretrained_backbone_weights: str | None = None
    resize_shape: int = 256
    crop_shape: tuple[int, int] | None = (224, 224)
    crop_is_random: bool = False
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 64
    cond_dim: int = 512 # the input dim of the condition

    # DiT layers
    is_tinyvla: bool = False
    noise_samples: int = 1
    time_as_cond: bool = True
    obs_as_cond: bool = True
    mlp_ratio: float = 4.0
    learn_sigma: bool = False

    # training
    num_train_timesteps: int = 100
    eval: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # inference
    num_inference_timesteps: int = 10
    num_queries: int = 16
    

    def __post_init__(self):
        # super().__post_init__()
        if self.model_size != "none":
            self.depth = MODEL_STRUCTURE[self.model_size]['depth'] # number of DiT blocks
            self.n_emb = MODEL_STRUCTURE[self.model_size]['n_emb'] # embedding size
            self.num_heads = MODEL_STRUCTURE[self.model_size]['num_heads'] 
            print(f'using model size {self.model_size} with depth {self.depth}, n_emb {self.n_emb}, num_heads {self.num_heads}')
        else:
            raise ValueError("model_size show not be 'none'")

        if not (self.vision_backbone.startswith("resnet") or self.vision_backbone.startswith("dino")):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        if self.model_size not in MODEL_STRUCTURE:
            raise ValueError(
                f"`model_size` must be one of {list(MODEL_STRUCTURE.keys())}. Got '{self.model_size}'."
            )
        

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        
        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

        
    # def get_optimizer_preset(self) -> AdamConfig:
    #     return AdamConfig(
    #         lr=self.optimizer_lr,
    #         betas=self.optimizer_betas,
    #         eps=self.optimizer_eps,
    #         weight_decay=self.optimizer_weight_decay,
    #     )
    
    # # def get_scheduler_preset(self) -> None:
    # #     return None
    # def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
    #     return DiffuserSchedulerConfig(
    #         name=self.scheduler_name,
    #         num_warmup_steps=self.scheduler_warmup_steps,
    #     )

    # @property
    # def observation_delta_indices(self) -> list:
    #     return list(range(1 - self.n_obs_steps, 1))

    # @property
    # def action_delta_indices(self) -> list:
    #     return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    # @property
    # def reward_delta_indices(self) -> None:
    #     return None
