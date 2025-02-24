###
 # @Author: Jiyuan Liu
 # @Date: 2025-02-15 14:30:57
 # @LastEditors: Jiyuan Liu
 # @LastEditTime: 2025-02-19 20:22:26
 # @FilePath: /fourier-lerobot-jy/resume.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by Fourier Intelligence Co. Ltd , All Rights Reserved. 
### 
python lerobot/scripts/train.py resume=True hydra.run.dir=/mnt/sda/lerobot/models/02-18-20-17_real_world_diffusion_pouring_arm_loss2_horizon64
