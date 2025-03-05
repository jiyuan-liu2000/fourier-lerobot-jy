#!/bin/bash
###
 # @Author: WenJiawei
 # @Date: 2025-03-04 15:08:29
 # @LastEditors: WenJiawei
 # @LastEditTime: 2025-03-05 15:29:02
 # @FilePath: /fourier-lerobot-jy/test_diffusion.sh
 # @Description: Script for testing diffusion policy
 # 
 # Copyright (c) 2025 by Fourier Intelligence Co. Ltd, All Rights Reserved. 
### 

# Default values
MODEL_PATH="/home/fourier/models/03-03-18-24_real_world_diffusion_pnp_coke_arm_loss2_horizon64_batch128_down4096/checkpoints/300000/pretrained_model"
ROOT="/home/fourier/data"
REPO_ID="final/fourier_pnp_coke"
EPISODE_IDX=5
MODE="dataset"

# Print test configuration
echo "Test Configuration:"
echo "Mode: $MODE"
echo "Model Path: $MODEL_PATH"
echo "Data Root: $ROOT"
echo "Repo ID: $REPO_ID"
echo "Episode Index: $EPISODE_IDX"

# Run the test
echo "Running test..."
python test_diffusion_policy.py \
    --mode "$MODE" \
    --root "$ROOT" \
    --repo-id "$REPO_ID" \
    --episode-idx "$EPISODE_IDX" \
    --model-path "$MODEL_PATH"

# Check if test was successful
if [ $? -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed!"
    exit 1
fi 