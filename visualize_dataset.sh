#!/bin/bash
###
 # @Author: WenJiawei
 # @Date: 2025-03-04 13:08:29
 # @LastEditors: WenJiawei
 # @LastEditTime: 2025-04-01 16:51:48
 # @FilePath: /fourier-lerobot-jy/visualize_dataset.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Fourier Intelligence Co. Ltd, All Rights Reserved. 
### 

# 设置默认值
DEFAULT_ROOT="/home/fourier/data"
DEFAULT_REPO="final/fourier_pnp_coke"
DEFAULT_EPISODE=0

# 解析命令行参数
DATA_ROOT=${1:-$DEFAULT_ROOT}
REPO_ID=${2:-$DEFAULT_REPO}
EPISODE_INDEX=${3:-$DEFAULT_EPISODE}

# 打印执行信息
echo "正在可视化数据集..."
echo "数据根目录: $DATA_ROOT"
echo "仓库ID: $REPO_ID"
echo "选择的片段: $EPISODE_INDEX"

# 执行可视化命令
python lerobot/scripts/visualize_dataset.py \
    --root "$DATA_ROOT" \
    --repo-id "$REPO_ID" \
    --episode-index "$EPISODE_INDEX"

# 检查命令执行状态
if [ $? -eq 0 ]; then
    echo "可视化完成！"
else
    echo "可视化过程中出现错误！"
    exit 1
fi 