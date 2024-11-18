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

from torchvision import transforms as v2

sys.path.append("../teleop")
from player import RealRGBDepthRobot

def transform_image(left_img, right_img):
    # import ipdb; ipdb.set_trace()
    # left_img = cv2.resize(left_img, (308, 224))
    # right_img = cv2.resize(right_img, (308, 224))

    left_img = left_img / 255.0
    right_img = right_img / 255.0

    left_img = left_img.view((1, 3, 240, 400)).to(device="cuda")
    right_img = right_img.view((1, 3, 240, 400)).to(device="cuda")
    qpos_data = qpos_data.view((1, 43)).to(device="cuda")

    patch_h = 16
    patch_w = 22
    transform = v2.Compose(
                [
                    # v2.ColorJitter(
                    #     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                    # ),
                    # v2.RandomPerspective(distortion_scale=0.5),
                    # v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    # v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)),
                    v2.Resize((patch_h * 14, patch_w * 14)),
                    # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                    # v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
    
    left_img = transform(left_img)
    right_img = transform(right_img)
    return left_img, right_img

if __name__ == "__main__":
    pretrained_policy_path = Path("outputs/train/2024-11-18/00-17-23_real_world_act_default/checkpoints/last")

    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
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
    policy.reset()
    player = RealRGBDepthRobot()

    step = 0
    done = False
    while not done:
        state, left_image, right_image = player.observe()
        image_to_show = left_image.copy()

        state = torch.from_numpy(state).to(torch.float32)
        left_image = torch.from_numpy(left_image).to(torch.float32).permute(2, 0, 1)
        right_image = torch.from_numpy(right_image).to(torch.float32).permute(2, 0, 1)

        left_image, right_image = transform_image(left_image, right_image)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        left_image = left_image.to(device, non_blocking=True)
        right_image = right_image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
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

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Step through the environment and receive a new observation
        if step < 3:
            # Avoid sudden movements in the first few steps
            player.step(numpy_action, image_to_show, mode="joint", time=1.5)
        else:
            player.step(numpy_action, image_to_show, mode="joint", time=0.0)

        step += 1

