###
 # @Author: Jiyuan Liu
 # @Date: 2025-02-15 14:30:57
 # @LastEditors: Jiyuan Liu
 # @LastEditTime: 2025-02-15 14:36:03
 # @FilePath: /fourier-lerobot-jy/resume.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by Fourier Intelligence Co. Ltd , All Rights Reserved. 
### 
python lerobot/scripts/train.py resume=True hydra.run.dir=outputs/train/2025-02-14/21-42-24_real_world_diffusion_pouring_obs16
