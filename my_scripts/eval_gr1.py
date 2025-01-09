"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""

from pathlib import Path

import imageio
import numpy as np
import torch
import sys
from huggingface_hub import snapshot_download

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

from torchvision import transforms as v2

from omegaconf import OmegaConf

sys.path.append("../teleop")
from player_gr2_dds import RealRGBDepthRobot

import h5py

sys.path.append("../")
from IL_Learning.vision.vlm.tracker import OnlineTracker
from IL_Learning.vision.vlm.masker import FindLLMOnline


def transform_image(left_img, right_img):
    left_img = left_img / 255.0
    right_img = right_img / 255.0
    left_img = left_img.reshape((3, 240, 400)).to(device="cuda")
    right_img = right_img.reshape((3, 240, 400)).to(device="cuda")

    patch_h = 16
    patch_w = 22
    transform = v2.Compose(
                [
                    # v2.Resize((patch_h * 14, patch_w * 14)),
                    v2.CenterCrop((patch_h * 14, patch_w * 14)),
                    # v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
    
    left_img = transform(left_img)
    right_img = transform(right_img)
    return left_img, right_img

if __name__ == "__main__":
    # pretrained_policy_path = Path("/home/dell/TeleVision/fourier-lerobot/outputs/train/2024-12-19/14-00-59_real_world_act_pick_and_place_30fps_2/checkpoints/040000/pretrained_model")
    pretrained_policy_path = Path('/home/dell/TeleVision/fourier-lerobot/outputs/train/2025-01-03/13-35-05_real_world_act_pick_and_place_left_new/checkpoints/060000/pretrained_model')
    # pretrained_policy_path = Path('/home/dell/TeleVision/fourier-lerobot/outputs/train/2025-01-07/10-55-52_real_world_diffusion_pick_and_place_left_new/checkpoints/115000/pretrained_model')
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    # policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        device = torch.device("cpu")
        print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")

    policy.to(device)
    
    # Reset the policy and environmens to prepare for rollout
    
    # policy.config.temporal_ensemble_coeff = None
    # policy.config.n_action_steps = 100 # 100
    # policy.config.n_action_steps = 4

    policy.reset()
    
    player = RealRGBDepthRobot(
                OmegaConf.load("my_scripts/gr2t5.yml")
            )
    tracker = OnlineTracker(
                            image_shape=(240, 400),
                            masker=FindLLMOnline, # Replacable with other maskers
                            # Arguments for initializing the masker
                            engine_path_decoder='../weights/mobile_sam_mask_decoder.engine',
                            engine_path_encoder='../weights/resnet18_image_encoder.engine',
                            base_url='http://192.168.6.129:8000/v1')
    player.set_tracker(tracker)
    player.reset_robot()
    player.capture_scene(target_objects=["drawer", "bottle on table", "left robot"])

    step = 0
    done = False
    ti = 0
    while not done:
        state, left_image, right_image = player.observe(mode="left", ti=ti)
        ti += 1
        state = torch.from_numpy(state).to(torch.float32)
        left_image = torch.from_numpy(left_image).to(torch.float32)
        right_image = torch.from_numpy(right_image).to(torch.float32)

        left_image, right_image = transform_image(left_image, right_image)
        image_to_show = left_image.squeeze(0).to("cpu").numpy()

        # Send data tensors from CPU to GPU
        # state_real = state_real.to(device, non_blocking=True)
        state = state.to(device, non_blocking=True)
        left_image = left_image.to(device, non_blocking=True)
        right_image = right_image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        # state_real = state_real.unsqueeze(0)
        state = state.unsqueeze(0)
        left_image = left_image.unsqueeze(0)
        right_image = right_image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.images.left": left_image,
            "observation.images.right": right_image,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)
        # print(action)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # # Step through the environment and receive a new observation
        if step < 2:
            # Avoid sudden movements in the first few steps
            player.step(numpy_action, image_to_show, mode="left", time=0.5)
        else:
            player.step(numpy_action, image_to_show, mode="left", time=0.0)

        step += 1

