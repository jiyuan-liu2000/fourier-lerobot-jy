 # @Author: Jiyuan Liu
###
 # @Author: WenJiawei
 # @Date: 2025-02-24 16:31:35
 # @LastEditors: WenJiawei
 # @LastEditTime: 2025-03-14 10:06:46
 # @FilePath: /fourier-lerobot-jy/train.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by Fourier Intelligence Co. Ltd, All Rights Reserved. 
### 
 # @Date: 2025-02-13 16:01:36
 # @LastEditors: Jiyuan Liu
 # @LastEditTime: 2025-02-13 16:01:49
 # @FilePath: /fourier-lerobot-jy/train.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by Fourier Intelligence Co. Ltd , All Rights Reserved. 
### 
python lerobot/scripts/train.py \
hydra.run.dir=/home/fourier/models/03-13-17-46_real_world_diffusion_pnp_coke_arm_loss2_horizon64_batch128_down4096_img112_224_loss_uncertainty/ \
resume=true 

