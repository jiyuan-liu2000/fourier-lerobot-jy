#!/bin/bash
###
 # @Author: WenJiawei
 # @Date: 2025-02-25 13:25:29
 # @LastEditors: WenJiawei
 # @LastEditTime: 2025-03-31 14:07:44
 # @FilePath: /fourier-lerobot-jy/run_raw2aloha2lerobot.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Fourier Intelligence Co. Ltd, All Rights Reserved. 
### 
###
 # @Author: WenJiawei
 # @Date: 2025-02-25 00:25:29
 # @LastEditors: WenJiawei
 # @LastEditTime: 2025-02-25 03:40:47
 # @FilePath: /fourier-lerobot-jy/run_raw2aloha.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Fourier Intelligence Co. Ltd, All Rights Reserved. 
### 

# 将原始数据转换为ALOHA格式
# hdf5_path: 原始数据的hdf5文件路径
# video_dir: 原始视频文件夹路径
# output_dir: 输出文件夹路径
# target_width/height: 目标图像尺寸为224x224
# start_key/end_key: 处理数据的起始和结束索引
# num_processes: 使用16个进程并行处理数据
python my_scripts/raw2aloha.py \
    --hdf5_path /home/fourier/data/raw/018_factory_12_27_coke_converted/trainable_data.hdf5 \
    --video_dir /home/fourier/data/raw/018_factory_12_27_coke_converted \
    --output_dir /home/fourier/data/processed/018_factory_12_27_coke_converted_processed \
    --target_width 224 \
    --target_height 224 \
    --start_key 0 \
    --end_key 183 \
    --num_processes 16


# 将ALOHA格式数据转换为LeRobot格式并上传到HuggingFace Hub
# raw-dir: ALOHA格式数据的路径
# local-dir: 转换后的LeRobot格式数据的本地保存路径
# raw-format: 原始数据格式为ALOHA HDF5
# force-override: 强制覆盖已有数据
# video: 是否包含视频数据
# repo-id: HuggingFace Hub上的仓库ID 强制参数
python lerobot/scripts/push_dataset_to_hub.py \
    --raw-dir /home/fourier/data/processed/018_factory_12_27_coke_converted_processed/ \
    --local-dir /home/fourier/data/final/fourier_pnp_coke \
    --raw-format aloha_hdf5 \
    --force-override 1 \
    --video 1 \
    --repo-id lerobot/aloha_static_pingpong_test