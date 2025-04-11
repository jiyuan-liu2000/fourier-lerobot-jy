#!/usr/bin/env python
"""
Script for validating DiffusionPolicy model
Using loaded images and randomly generated states as input
"""

import torch
import numpy as np
from pathlib import Path
import cv2
from torchvision import transforms
import logging
import time
import matplotlib.pyplot as plt
from datetime import datetime
import torch.utils.tensorboard as tb
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lerobot.common.datasets.factory import make_dataset, resolve_delta_timestamps
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize
from pprint import pformat
import sys
import matplotlib.cm as cm

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.scaledp.modeling_scaledp import ScaleDPPolicy

def transform_image(image, crop_size=(224, 224), resize_size=256):
    """Preprocess image"""
    # Convert to float and normalize
    image = image.astype(np.float32) / 255.0
    
    # Convert to torch tensor and adjust channel order
    image = torch.from_numpy(image).permute(2, 0, 1)
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
    ])
    
    image = transform(image)
    return image

def generate_random_state(state_dim=7):
    """Generate random state vector"""
    return np.random.uniform(-1, 1, size=state_dim)

def load_policy(policy_path):
    """Load pretrained policy"""
    # policy = DiffusionPolicy.from_pretrained(policy_path)
    policy = ScaleDPPolicy.from_pretrained(policy_path)

    policy.eval()  # Set to evaluation mode
    return policy

def visualize_action_trajectory(actions, save_path):
    """Visualize action trajectories for both arms and hands"""
    plt.figure(figsize=(20, 10))
    
    # Split actions into left/right arm and left/right hand
    left_arm_actions = actions[:, :7]    # First 7 dims are left arm
    right_arm_actions = actions[:, 7:14]  # Next 7 dims are right arm
    left_hand_actions = actions[:, 14:20]  # Next 6 dims are left hand
    right_hand_actions = actions[:, 20:]  # Last 6 dims are right hand
    
    # Plot left arm actions
    plt.subplot(2, 2, 1)
    for i in range(left_arm_actions.shape[1]):
        plt.plot(left_arm_actions[:, i], label=f'Joint {i+1}')
    plt.title('Left Arm Joint Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Joint Angles')
    plt.legend()
    plt.grid(True)
    
    # Plot right arm actions
    plt.subplot(2, 2, 2)
    for i in range(right_arm_actions.shape[1]):
        plt.plot(right_arm_actions[:, i], label=f'Joint {i+1}')
    plt.title('Right Arm Joint Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Joint Angles')
    plt.legend()
    plt.grid(True)
    
    # Plot left hand actions
    plt.subplot(2, 2, 3)
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'Thumb Lateral']
    for i in range(left_hand_actions.shape[1]):
        plt.plot(left_hand_actions[:, i], label=finger_names[i])
    plt.title('Left Hand Joint Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Joint Angles')
    plt.legend()
    plt.grid(True)
    
    # Plot right hand actions
    plt.subplot(2, 2, 4)
    for i in range(right_hand_actions.shape[1]):
        plt.plot(right_hand_actions[:, i], label=finger_names[i])
    plt.title('Right Hand Joint Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Joint Angles')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Add a text box with action dimensions explanation
    plt.figtext(0.02, 0.02, 
                'Action dimensions:\n' +
                'Left Arm: dims 0-6 (7 joints)\n' +
                'Right Arm: dims 7-13 (7 joints)\n' +
                'Left Hand: dims 14-19 (5 fingers + thumb lateral)\n' +
                'Right Hand: dims 20-25 (5 fingers + thumb lateral)',
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=8)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def print_policy_config(policy):
    """Print complete policy configuration"""
    config = policy.config
    logging.info("\nComplete Policy Configuration:")
    logging.info("-" * 50)
    
    # Get all attributes of the config object
    config_dict = vars(config)
    
    # Print all configuration parameters
    for key, value in sorted(config_dict.items()):
        logging.info(f"{key}: {value}")
    
    logging.info("-" * 50 + "\n")

def load_dataset(repo_id: str, root: str, episode_idx: int = 0):
    """Load a specific episode from lerobot dataset
    
    Args:
        repo_id: Dataset repository ID (e.g. 'final/fourier_pnp_coke')
        root: Root directory for dataset
        episode_idx: Episode index to test
    """
    # Load dataset with root path
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        split="train"  # Use training split by default
    )
    
    # Print dataset info
    print(f"\n{dataset[0]['observation.image.left'].shape=}")  # (4,c,h,w)
    print(f"{dataset[0]['observation.state'].shape=}")  # (8,c)
    print(f"{dataset[0]['action'].shape=}\n")  # (64,c)

    # Get frame indices for specific episode
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    indices = list(range(from_idx, to_idx))
    print(f"Episode {episode_idx} frame range: {from_idx} to {to_idx}")
    print(f"Found {len(indices)} frames")
    
    # Create test dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=32,
        shuffle=True,
    )
    for batch in dataloader:
        print(f"{batch['observation.image.left'].shape=}")  # torch.Size([32, 3, 224, 224])
        print(f"{batch['observation.state'].shape=}")  # torch.Size([32, 26])
        print(f"{batch['action'].shape=}")  # torch.Size([32, 26])
        break
    
    return dataset, indices

def visualize_step_data(image, state, action=None, target_action=None, step=0, episode_idx=0, state_history=None, action_history=None):
    """Visualize current step's input data and actions in a single window"""
    # Convert image tensor to numpy array for visualization
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # 获取最大维度以决定颜色数量
    max_dims = max(
        len(state_history['values'][0]) if state_history else 0,
        len(action_history['predicted'][0]) if action_history else 0
    )
    colors = cm.tab20(np.linspace(0, 1, max_dims))
    
    # 定义关节名称
    left_arm_joints = ["L shoulder pitch", "L shoulder roll", "L shoulder yaw", 
                       "L elbow pitch", "L elbow yaw", "L wrist pitch", "L wrist roll"]
    right_arm_joints = ["R shoulder pitch", "R shoulder roll", "R shoulder yaw", 
                        "R elbow pitch", "R elbow yaw", "R wrist pitch", "R wrist roll"]
    left_hand_joints = ["L thumb", "L index", "L middle", "L ring", "L pinky", "L thumb lateral"]
    right_hand_joints = ["R thumb", "R index", "R middle", "R ring", "R pinky", "R thumb lateral"]
    
    joint_names = left_arm_joints + right_arm_joints + left_hand_joints + right_hand_joints
    
    # Create figure with subplots
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(24, 12)
    
    # Create grid for subplots with adjusted ratios
    gs = plt.GridSpec(3, 2, 
                     height_ratios=[1.2, 1.2, 1.2],
                     width_ratios=[1.2, 2.5],
                     hspace=0.3,
                     wspace=0.3)
    
    # Plot image and current action comparison on the left
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title(f"Input Image\n(Episode {episode_idx}, Step {step})", pad=10)
    ax1.axis('off')
    
    # Plot current action comparison below image
    if action is not None and target_action is not None:
        ax2 = plt.subplot(gs[1:, 0])
        action_np = action.cpu().numpy() if torch.is_tensor(action) else action
        target_np = target_action.cpu().numpy() if torch.is_tensor(target_action) else target_action
        
        x = np.arange(len(action_np))
        width = 0.35
        
        # 计算误差
        error = np.abs(action_np - target_np)
        
        # 绘制柱状图
        pred_bars = ax2.bar(x - width/2, action_np, width, label='Current Predicted', color='royalblue')
        target_bars = ax2.bar(x + width/2, target_np, width, label='Current Target', color='darkorange')
        
        # 为每个关节添加误差值标注
        for i, (pred, target, err) in enumerate(zip(action_np, target_np, error)):
            # 确定标注位置（较高的柱子上方）
            y_pos = max(pred, target) + 0.1
            
            # 设置标注格式（误差较大的用红色）
            color = 'red' if err > np.percentile(error, 75) else 'black'
            fontweight = 'bold' if err > np.percentile(error, 75) else 'normal'
            
            # 添加标注
            ax2.annotate(f"{err:.2f}",
                        xy=(i, y_pos),
                        ha='center',
                        va='bottom',
                        fontsize=7,
                        color=color,
                        fontweight=fontweight)
        
        # 标记关节名称
        if len(action_np) <= 10:  # 如果关节较少，直接显示全部标签
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{i}" for i in range(len(action_np))], rotation=45, ha="right")
        else:  # 如果关节较多，只显示主要关节
            key_indices = [0, 7, 14, 20]  # 左臂开始、右臂开始、左手开始、右手开始
            ax2.set_xticks(key_indices)
            ax2.set_xticklabels(["L Arm", "R Arm", "L Hand", "R Hand"], rotation=45, ha="right")
            
            # 添加垂直线分隔不同组件
            for idx in key_indices[1:]:
                ax2.axvline(x=idx-0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 标记误差最大的关节
        top_errors = np.argsort(error)[-3:]  # 获取误差最大的3个关节
        for i in top_errors:
            ax2.annotate(f"{joint_names[i]}",
                        xy=(i, max(action_np[i], target_np[i]) + 0.3),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        ax2.set_title("Current Step\nAction Comparison", pad=10)
        ax2.set_xlabel("Action Dimension")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)
    
    # Plot state history with adjusted height
    ax3 = plt.subplot(gs[0, 1])
    if state_history is not None:
        for i in range(len(state_history['values'][0])):
            ax3.plot(state_history['steps'], 
                    [states[i] for states in state_history['values']], 
                    label=f'State {i}',
                    color=colors[i % len(colors)])
    ax3.set_title("State History", pad=10)
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("State Value")
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    ax3.grid(True)
    
    # Plot predicted action history
    ax4 = plt.subplot(gs[1, 1])
    if action_history is not None:
        for i in range(len(action_history['predicted'][0])):
            label = joint_names[i] if i < len(joint_names) else f'Action {i}'
            ax4.plot(action_history['steps'],
                    [actions[i] for actions in action_history['predicted']],
                    label=label,
                    color=colors[i % len(colors)])
    ax4.set_title("Predicted Action History", pad=10)
    ax4.set_xlabel("Time Steps")
    ax4.set_ylabel("Action Value")
    ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    ax4.grid(True)
    
    # Plot target action history
    ax5 = plt.subplot(gs[2, 1])
    if action_history is not None:
        for i in range(len(action_history['target'][0])):
            label = joint_names[i] if i < len(joint_names) else f'Action {i}'
            ax5.plot(action_history['steps'],
                    [actions[i] for actions in action_history['target']],
                    label=label,
                    color=colors[i % len(colors)])
    ax5.set_title("Target Action History", pad=10)
    ax5.set_xlabel("Time Steps")
    ax5.set_ylabel("Action Value")
    ax5.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    ax5.grid(True)
    
    # Align the x-axes of all history plots
    if action_history is not None:
        xlim = (min(action_history['steps']), max(action_history['steps']))
        ax3.set_xlim(xlim)
        ax4.set_xlim(xlim)
        ax5.set_xlim(xlim)
    
    plt.tight_layout()
    plt.pause(0.01)

def test_with_dataset(policy, dataset, indices, device, output_dir, episode_idx, step_vis=False):
    """Test policy with real dataset samples
    
    Args:
        policy: The policy to test
        dataset: Dataset to test on
        indices: Frame indices for the episode
        device: Device to run inference on
        output_dir: Directory to save results
        episode_idx: Episode index being tested
        step_vis: Flag to enable step-by-step visualization
    """
    # Set up tensorboard
    writer = tb.SummaryWriter(output_dir / "tensorboard")
    
    # Create dataloader for single episode
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=indices,
        num_workers=0,
        pin_memory=True
    )
    
    all_actions = []
    all_targets = []
    total_loss = 0
    n_steps = len(indices)
    
    # Initialize histories for visualization
    state_history = {'steps': [], 'values': []} if step_vis else None
    action_history = {
        'steps': [],
        'predicted': [],
        'target': []
    } if step_vis else None
    
    # Create figure for visualization
    if step_vis:
        plt.figure(figsize=(20, 12))
    
    try:
        start_time = time.time()
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Process observation for policy
            observation = {
                "observation.state": batch["observation.state"],
                "observation.image.left": batch["observation.image.left"],
            }
            # import pdb; pdb.set_trace()
            # Generate action using policy
            with torch.inference_mode():
                action = policy.select_action(observation)
            
            if step_vis:
                # Update state history
                current_state = batch["observation.state"][0].cpu().numpy()
                state_history['steps'].append(step)
                state_history['values'].append(current_state)
                
                # Update action history
                action_history['steps'].append(step)
                action_history['predicted'].append(action[0].cpu().numpy())
                action_history['target'].append(batch["action"][0].cpu().numpy())
                
                # Visualize current step with all data
                visualize_step_data(
                    image=batch["observation.image.left"][0],
                    state=batch["observation.state"][0],
                    action=action[0],
                    target_action=batch["action"][0],
                    step=step,
                    episode_idx=episode_idx,
                    state_history=state_history,
                    action_history=action_history
                )
                
                # Wait for window close or key press
                plt.waitforbuttonpress()
            
            # Get target action from dataset
            target_action = batch["action"]
            
            # Calculate MSE loss
            # 反归一化后的动作值计算的损失
            loss = F.mse_loss(action, target_action)
            total_loss += loss.item()
            
            # Log detailed metrics to tensorboard
            writer.add_scalar("Loss/mse", loss.item(), step)
            writer.add_scalar("Performance/fps", (step + 1) / (time.time() - start_time), step)
            
            # Log images
            writer.add_images("Images/input", batch["observation.image.left"], step)
            
            # Log actions and targets separately for each joint
            for i in range(action.shape[1]):
                writer.add_scalars(f"Actions/Joint_{i}", {
                    'predicted': action[0, i].item(),
                    'target': target_action[0, i].item()
                }, step)
            
            # Log state values
            for i in range(batch["observation.state"].shape[1]):
                writer.add_scalar(f"States/Dim_{i}", batch["observation.state"][0, i].item(), step)
            
            # Calculate and log error metrics
            abs_error = torch.abs(action - target_action)
            writer.add_scalar("Metrics/max_abs_error", abs_error.max().item(), step)
            writer.add_scalar("Metrics/mean_abs_error", abs_error.mean().item(), step)
            
            # Store for final visualization
            numpy_action = action.cpu().numpy()
            numpy_target = target_action.cpu().numpy()
            all_actions.append(numpy_action)
            all_targets.append(numpy_target)
            
            # Print progress
            if step % 10 == 0:
                elapsed_time = time.time() - start_time
                fps = (step + 1) / elapsed_time
                avg_loss = total_loss / (step + 1)
                logging.info(f"Step: {step}, FPS: {fps:.2f}, Avg Loss: {avg_loss:.4f}")
                
    except Exception as e:
        logging.error(f"Error during testing: {e}")
        raise
        
    finally:
        if all_actions:
            # Calculate final statistics
            avg_loss = total_loss / n_steps
            logging.info(f"\nFinal Statistics:")
            logging.info(f"Average MSE Loss: {avg_loss:.4f}")
            
            # Log final metrics
            writer.add_hparams(
                {"episode_idx": episode_idx},
                {
                    "avg_mse_loss": avg_loss,
                    "total_frames": n_steps,
                    "avg_fps": n_steps / (time.time() - start_time)
                }
            )
            
            # Visualize trajectories
            all_actions = np.concatenate(all_actions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            visualize_action_trajectory(all_actions, output_dir / "predicted_trajectory.png")
            visualize_action_trajectory(all_targets, output_dir / "target_trajectory.png")
            
            # Save final statistics
            with open(output_dir / "test_stats.txt", "w") as f:
                f.write(f"Total Steps: {n_steps}\n")
                f.write(f"Average MSE Loss: {avg_loss:.4f}\n")
        
        writer.close()
        
        # Print tensorboard viewing instructions
        logging.info(f"\nTo view results in tensorboard, run:")
        logging.info(f"tensorboard --logdir {output_dir}")
        logging.info("Then open http://localhost:6006 in your browser")

        if step_vis:
            plt.close()

def setup_logging(output_dir):
    """Setup logging to both file and console"""
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # Setup file handler
    file_handler = logging.FileHandler(output_dir / "test_log.txt")
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def main():
    # Create output directory first
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"test_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    # Add argument parser for test mode
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "dataset", "recorded_data"], default="random",
                       help="Test mode: random state or dataset")
    parser.add_argument("--root", type=str, default="/home/fourier/data",
                       help="Root directory for dataset")
    parser.add_argument("--repo-id", type=str, default="final/fourier_pnp_coke",
                       help="Dataset repository ID")
    parser.add_argument("--episode-idx", type=int, default=0,
                       help="Episode index to test")
    parser.add_argument("--model-path", type=str, 
                       default="/home/fourier/models/03-03-18-24_real_world_diffusion_pnp_coke_arm_loss2_horizon64_batch128_down4096/checkpoints/300000/pretrained_model",
                       help="Path to pretrained model")
    parser.add_argument("--step-vis", action="store_true",
                       help="Enable step-by-step visualization")
    parser.add_argument("--record-dir", type=str, default="/home/fourier/fourier-lerobot-jy/record_data/",
                       help="Path to recorded data")
        
    args = parser.parse_args()
    
    # Log command line arguments
    logger.info("Test Configuration:")
    logger.info("-" * 50)
    logger.info(pformat(vars(args)))
    logger.info("-" * 50)
    
    # Load pretrained model
    policy_path = Path(args.model_path)
    policy = load_policy(policy_path)
    
    # Log model configuration
    # logger.info("\nModel Configuration:")
    # logger.info("-" * 50)
    # logger.info(pformat(OmegaConf.to_container(policy.config)))
    # logger.info("-" * 50)
    
    # Save complete configuration to file
    with open(output_dir / "model_config.txt", "w") as f:
        f.write("Test Configuration:\n")
        f.write("-" * 50 + "\n")
        f.write(pformat(vars(args)) + "\n\n")
        f.write("Model Configuration:\n")
        f.write("-" * 50 + "\n")
        f.write(pformat(vars(policy.config)) + "\n")
        f.write("-" * 50 + "\n")
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for inference")
    
    policy.to(device)
    policy.reset()
    
    # Load test image
    image_path = "/home/fourier/Pictures/Figure_1.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save original image
    plt.imsave(output_dir / "input_image.png", image)
    
    # Set test parameters
    n_steps = 100  # Number of test steps
    state_dim = 26  # State vector dimension
    
    # Store all actions for visualization
    all_actions = []
    
    if args.mode == "dataset":
        # Load dataset using repo_id and root
        dataset, indices = load_dataset(
            repo_id=args.repo_id,
            root=args.root,
            episode_idx=args.episode_idx
        )
        
        # Test with dataset
        test_with_dataset(
            policy=policy,
            dataset=dataset,
            indices=indices,
            device=device,
            output_dir=output_dir,
            episode_idx=args.episode_idx,
            step_vis=args.step_vis  # Pass visualization flag
        )

    elif args.mode == "recorded_data":
        dataset, indices = load_dataset(
            repo_id=args.repo_id,
            root=args.root,
            episode_idx=args.episode_idx
        )

        test_with_recorded_data(
            policy=policy,
            record_dir=args.record_dir,
            device=device,
            output_dir=output_dir,
            dataset=dataset,
            indices=indices
        )

    else:
        try:
            start_time = time.time()
            for step in range(n_steps):
                # Prepare input data
                # Process image
                left_image = transform_image(image)
                left_image = left_image.unsqueeze(0).to(device)  # Add batch dimension
                
                # Generate random state
                state = generate_random_state(state_dim)
                state = torch.from_numpy(state).float()
                state = state.unsqueeze(0).to(device)  # Add batch dimension
                
                # Build observation dictionary
                observation = {
                    "observation.state": state,
                    "observation.image.left": left_image,
                }
                
                # Generate action using policy
                with torch.inference_mode():
                    action = policy.select_action(observation)
                
                # Get numpy format action
                numpy_action = action.squeeze(0).cpu().numpy()
                all_actions.append(numpy_action)
                
                # Print progress and action
                if step % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps = (step + 1) / elapsed_time
                    logger.info(f"Step: {step}, FPS: {fps:.2f}")
                    logger.info(f"Action: {numpy_action}")
                    
                    # Save processed image
                    processed_image = left_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    plt.imsave(output_dir / f"processed_image_step_{step}.png", 
                              np.clip(processed_image, 0, 1))
                
        except Exception as e:
            logger.error(f"Error during testing: {e}")
            raise
        
        finally:
            # Visualize action trajectories
            all_actions = np.stack(all_actions)
            visualize_action_trajectory(all_actions, output_dir / "action_trajectory.png")
            
            # Calculate and print performance statistics
            total_time = time.time() - start_time
            avg_fps = n_steps / total_time
            logger.info(f"\nPerformance Statistics:")
            logger.info(f"Total Steps: {n_steps}")
            logger.info(f"Total Time: {total_time:.2f} seconds")
            logger.info(f"Average FPS: {avg_fps:.2f}")
            
            # Save performance statistics to file
            with open(output_dir / "performance_stats.txt", "w") as f:
                f.write(f"Total Steps: {n_steps}\n")
                f.write(f"Total Time: {total_time:.2f} seconds\n")
                f.write(f"Average FPS: {avg_fps:.2f}\n")




def test_with_recorded_data(policy, record_dir, device, output_dir, dataset, indices):
    """使用录制的数据进行测试
    
    Args:
        policy: 要测试的策略模型
        record_dir: 录制数据的目录路径
        device: 运行设备
        output_dir: 输出目录
    """
    import json
    # 加载录制的数据
    record_dir = Path(record_dir)
    
    # 加载状态数据
    with open(record_dir / "state_data.json", "r") as f:
        state_data = json.load(f)
    
    # 设置视频捕获
    video_path = str(record_dir / "000000000.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 尝试备用文件名
        video_path = str(record_dir / "000000000.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 创建tensorboard writer
    writer = tb.SummaryWriter(output_dir / "tensorboard")
    
    # 存储所有动作用于可视化
    all_actions = []
    
    # 创建动作历史数据存储 - 用于绘制时间序列曲线
    action_history = {
        'timestamps': [],
        'values': []
    }
    
    # 定义关节名称（用于图例）
    joint_names = [
        "左肩pitch", "左肩roll", "左肩yaw", "左肘", "左腕roll", "左腕pitch", "左腕yaw",
        "右肩pitch", "右肩roll", "右肩yaw", "右肘", "右腕roll", "右腕pitch", "右腕yaw"
    ]
    if len(joint_names) < 26:  # 如果实际关节数量超过预定义名称
        for i in range(len(joint_names), 26):
            joint_names.append(f"Joint {i}")

    # 保存第一帧变换前后的图像
    if_save_first_frame = True

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=indices,
        num_workers=0,
        pin_memory=True
    )

    # 定义数据源类型枚举
    class DataSource:
        RECORDED_DATA = "recorded_data"  # 使用录制的数据
        DATASET = "dataset"              # 使用数据集
        REAL_TIME = "real_time"          # 实时数据
        
    # 设置当前使用的数据源
    use_which_data_source = DataSource.RECORDED_DATA
    
    # 创建固定大小的图形窗口
    plt.figure(figsize=(16, 8))
    
    # 设置最大历史长度
    max_history_length = 50  # 最多显示50帧的历史数据

    try:
        start_time = time.time()
        frame_idx = 0

        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Process observation for policy
            observation = {
                "observation.state": batch["observation.state"],
                "observation.image.left": batch["observation.image.left"],
            }



            ret, frame = cap.read()
            if not ret:
                break
                
            # 获取对应的状态数据
            if frame_idx >= len(state_data):
                break
            current_state = state_data[frame_idx]["state"]
            
            # 调整BGR到RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理图像
            if if_save_first_frame:
                raw_image = frame.copy()
                plt.imsave(output_dir / f"frame_{frame_idx}.png", raw_image)

            left_image = transform_image(frame)

            if if_save_first_frame:
                processed_image = left_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                plt.imsave(output_dir / f"processed_image_step_{frame_idx}.png", 
                        processed_image)
                if_save_first_frame = False

            left_image = left_image.unsqueeze(0).to(device)
            
            # 处理状态
            state = torch.tensor(current_state, dtype=torch.float32)
            state = state.unsqueeze(0).to(device)
            
            # 构建观测字典
            observation_from_recorded_data = {
                "observation.state": state,
                "observation.image.left": left_image,
            }

            # import pdb; pdb.set_trace()
            
            # observation["observation.state"] = observation_from_recorded_data["observation.state"]
            observation["observation.image.left"] = observation_from_recorded_data["observation.image.left"]
            if frame_idx == 0:
                observation["observation.state"] = torch.zeros(1, 26, dtype=torch.float32).to(device)
                observation["observation.state"] = torch.tensor([[-0.05, 0.071, 0.091, -1.3767406940460205, 0.2229975163936615, -0.1671057641506195, -0.1857181191444397, -0.20743083953857422, -0.06330453604459763, 0.11242031306028366, -1.2058242559432983, 0.06342902779579163, -0.023585783317685127, -0.22980284690856934, 0.12083300203084946, 0.15208299458026886, 0.16249999403953552, 0.07916700094938278, 5.84375, 1.5791670083999634, 0.3760420083999634, 0.3656249940395355, 0.37083300948143005, 0.5843750238418579, 5.5416669845581055, 6.209374904632568]], dtype=torch.float32).to(device)
            # 打印图像形状和值的比较
            print("\n图像对比:")
            print(f"left_image shape: {left_image.shape}")
            print(f"batch image shape: {batch['observation.image.left'].shape}")
            print(f"left_image value range: [{left_image.min():.3f}, {left_image.max():.3f}]")
            print(f"batch image value range: [{batch['observation.image.left'].min():.3f}, {batch['observation.image.left'].max():.3f}]")
            
            # 计算差异
            diff = (left_image - batch["observation.image.left"]).abs()
            print(f"平均绝对差异: {diff.mean():.3f}")
            print(f"最大绝对差异: {diff.max():.3f}")


            # 使用策略生成动作
            with torch.inference_mode():
                action = policy.select_action(observation)
            
            # 获取numpy格式的动作
            numpy_action = action.squeeze(0).cpu().numpy()
            all_actions.append(numpy_action)
            
            # 更新动作历史数据
            action_history['timestamps'].append(frame_idx)
            action_history['values'].append(numpy_action)
            
            # 限制历史数据长度
            if len(action_history['timestamps']) > max_history_length:
                action_history['timestamps'] = action_history['timestamps'][-max_history_length:]
                action_history['values'] = action_history['values'][-max_history_length:]
            
            # 记录到tensorboard
            if frame_idx % 10 == 0:
                elapsed_time = time.time() - start_time
                fps = (frame_idx + 1) / elapsed_time
                writer.add_scalar("Performance/fps", fps, frame_idx)
                writer.add_images("Images/input", left_image, frame_idx)
                
                # 记录动作
                for i in range(action.shape[1]):
                    writer.add_scalar(f"Actions/Joint_{i}", 
                                    numpy_action[i], frame_idx)
                
                # 记录状态值
                for i in range(state.shape[1]):
                    writer.add_scalar(f"States/Dim_{i}", 
                                    current_state[i], frame_idx)
                
                logging.info(f"Frame: {frame_idx}, FPS: {fps:.2f}")
            
            # 仅使用单一图表显示左臂关节角度曲线
            plt.clf()  # 清除当前图形
            
            # 定义颜色映射，为不同关节分配不同颜色
            colors = plt.cm.tab10(np.linspace(0, 1, 7))  # 只使用7种颜色
            
            # 只绘制左臂关节曲线(0-6)
            joints_to_plot = range(min(7, len(numpy_action)))
            
            for i in joints_to_plot:
                # 提取该关节的历史值
                joint_history = [values[i] for values in action_history['values']]
                plt.plot(action_history['timestamps'], joint_history, 
                        label=f"Joint {i}", color=colors[i], 
                        linewidth=2, alpha=0.8)
            
            # 标记当前时间点
            plt.axvline(x=frame_idx, color='r', linestyle='--', alpha=0.5)
            
            plt.title(f"Left Arm Joint Angles - Frame {frame_idx}")
            plt.xlabel("Frame")
            plt.ylabel("Joint Angle")
            plt.legend(fontsize='small')
            plt.grid(True, alpha=0.3)
            
            # 设置x轴范围，显示最近的历史
            if len(action_history['timestamps']) > 0:
                min_time = max(0, frame_idx - max_history_length)
                plt.xlim(min_time, frame_idx + 5)
            
            # 显示当前帧的关节值
            if len(numpy_action) >= 7:
                left_arm_values = numpy_action[:7]
                value_text = "Current values: " + ", ".join([f"{v:.3f}" for v in left_arm_values])
                plt.figtext(0.5, 0.01, value_text, ha='center', fontsize=9, 
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
            frame_idx += 1
            
    except Exception as e:
        logging.error(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        cap.release()
        
        if all_actions:
            # 计算最终统计信息
            logging.info(f"\n最终统计:")
            logging.info(f"总帧数: {frame_idx}")
            
            # 记录最终指标
            writer.add_hparams(
                {"test_mode": "recorded_data"},
                {
                    "total_frames": frame_idx,
                    "avg_fps": frame_idx / (time.time() - start_time)
                }
            )
            
            # 可视化动作轨迹
            all_actions = np.stack(all_actions)
            visualize_action_trajectory(all_actions, 
                                     output_dir / "predicted_trajectory.png")
            
        writer.close()


if __name__ == "__main__":
    main() 