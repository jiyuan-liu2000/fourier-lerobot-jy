
 * @Author: WenJiawei
 * @Date: 2025-02-24 03:15:03
 * @LastEditors: WenJiawei
 * @LastEditTime: 2025-02-25 03:57:15
 * @FilePath: /fourier-lerobot-jy/README.md
 * @Description: 
 * 
 * Copyright (c) 2025 by Fourier Intelligence Co. Ltd, All Rights Reserved. 
---

## 调试日志

本节记录代码库的各种调试更改和优化。

### 图像处理优化 (2025-02-24)

对图像处理流程进行改进，确保数据格式一致性，并实现高效的图像压缩存储。

#### 原始图像处理流程
```python
resized_frame = maintain_aspect_ratio_resize(frame, target_size)
resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
transposed_frame = np.transpose(resized_frame, (2, 0, 1))
frames.append(transposed_frame)
```

#### 更新后的图像处理流程
```python
resized_frame = maintain_aspect_ratio_resize(frame, target_size)
resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
transposed_frame = np.transpose(resized_frame, (2, 0, 1))
# 转回HWC格式用于压缩
transposed_frame = np.transpose(transposed_frame, (1, 2, 0))
transposed_frame = transposed_frame.astype(np.uint8)
_, transposed_frame = cv2.imencode('.jpg', transposed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
frames.append(transposed_frame)
```

#### 处理流程详情
1. 图像预处理：
   - 调整图像尺寸（maintain_aspect_ratio_resize）
   - 转换颜色空间（BGR转RGB）

2. 格式转换(可省略)：
   - 转换为CHW格式（通道，高度，宽度）
   - 转回HWC格式（高度，宽度，通道）

3. 图像压缩 (可省略)：
   - 确保数据类型为uint8
   - 使用JPEG压缩（质量95）

#### 数据格式流程
```
原始图像 (HWC, BGR)
  ↓
调整尺寸 (224x224x3, BGR)
  ↓
颜色转换 (224x224x3, RGB)
  ↓
转置为CHW (3x224x224, RGB)
  ↓
转置为HWC (224x224x3, RGB)
  ↓
JPEG压缩 (压缩后的字节数据)
```

#### 重要说明
1. 在进行JPEG压缩前确保图像格式正确（HWC）
2. 压缩质量设置为95，平衡质量和大小
3. 数据类型必须是uint8才能正确压缩

#### 效果
1. 减小数据存储空间
2. 保持图像质量（95%压缩质量）
3. 与数据加载代码保持兼容性
4. 减小数据加载时间

---


#### 数据增强修复 (2025-02-26)

在训练过程中，发现数据增强部分出现了错误，主要是由于`sharpness`变换的问题。通过以下修改解决：

1. 将配置文件中的`sharpness.weight`设置为0，禁用该增强方式
2. 保留其他数据增强方法（亮度、对比度、饱和度、色调、高斯模糊）

#### Diffusion Policy代码注释

为了更好地理解扩散策略的实现，为`modeling_diffusion.py`添加了详细的中文注释，主要包括：

1. 核心类结构说明
   - `DiffusionPolicy`: 主策略类，处理输入归一化和动作选择
   - `DiffusionModel`: 扩散模型实现，包含噪声预测网络
   - `DiffusionConditionalUnet1d`: 一维条件UNet网络

2. 关键组件注释
   - 时间步编码器（SinusoidalPosEmb）
   - FiLM条件调制机制
   - 残差卷积块实现
   - 空间Softmax特征提取

3. 训练和推理流程
   - 噪声添加和去噪过程
   - 条件生成机制
   - 动作轨迹预测

这些注释有助于理解扩散策略的工作原理，特别是其如何将视觉特征与状态信息结合，生成平滑的机器人动作轨迹。

---
#### 调试日志 (2025-02-28)

进行上机测试，流程为初始位置上电，初始化后摆放至开始位置，然后进行测试。

测试循环帧率为20Hz

传输文件命令为：
```bash
scp -r root@192.168.1.100:/home/root/lerobot/data/ /home/wenjiawei/fourier-lerobot-jy/data/
scp -2r 100000/  ubuntu@192.168.12.166:/mnt/sda/data/models/wjw_0304/
```
eval文件中关键代码为：
```python

def load_policy(policy_path, policy_mode="act"):
    pretrained_policy_path = Path(policy_path)
    if policy_mode == "act":
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    elif policy_mode == "dp":
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    else:
        raise ValueError(
            f"Invalid policy mode input: {policy_mode}, only act or dp mode support"
        )
    policy.n_action_steps = 8
    policy.eval()
    return policy


if __name__ == "__main__":
    # Load the pretrained policy
    policy = load_policy(
        "/mnt/sda/data/models/02-26-14-55_real_world_diffusion_pnp_coke_arm_loss2_horizon64/checkpoints/300000/pretrained_model",
        policy_mode="dp",
    )

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        device = torch.device("cpu")
        print(
            f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU."
        )

    policy.to(device)
    policy.reset()

    # Initialize the GR2 player
    cam = OakCamera(rgb_resolution=(640, 480), depth_resolution=(640, 480), fps=30)
    player = GR1Player(
        OmegaConf.load("controller/configs/gr1t2_upper_body.yaml"), camera=cam
    )
    # Move robot to the initial position
    player.reset_robot()

    step = 0
    done = False
    vis_img = True
    try:
        start_time = time.time()
        while not done:
            state, left_image = player.observe(mode="bimanual")
            state = torch.from_numpy(state).to(torch.float32)
            logging.debug(f"get observation img shape :{left_image.shape}")
            logging.debug(f"get observation state shape :{state.shape}")

            left_image = transform_image(left_image, crop_size=(224, 224))
            logging.debug(f"input net img shape:{left_image.shape}")

            if vis_img:
                image_to_show = left_image.squeeze(0).to("cpu").numpy()
            else:
                image_to_show = None

            # Send data tensors from CPU to GPU
            state = state.to(device, non_blocking=True)
            left_image = left_image.to(device, non_blocking=True)

            # Add extra (empty) batch dimension, required to forward the policy
            state = state.unsqueeze(0)
            left_image = left_image.unsqueeze(0)

            # Create the policy input dictionary
            observation = {
                "observation.state": state,
                "observation.image.left": left_image,
            }

            # Predict the next action with respect to the current observation
            with torch.inference_mode():
                action = policy.select_action(observation)

            # Prepare the action for the environment
            numpy_action = action.squeeze(0).to("cpu").numpy()

            # # Step through the environment and receive a new observation
            if step < 2:
                # Avoid sudden movements in the first few steps
                player.step(numpy_action, image_to_show, mode="bimanual", time=0.5)
            else:
                player.step(numpy_action, image_to_show, mode="bimanual", time=0.0)

            step += 1
            # Calculate the frame rate
            if step % 10 == 0:
                elapsed_time = time.time() - start_time
                frame_rate = step / elapsed_time
                logging.info(f"Step: {step}, Frame Rate: {frame_rate:.2f} FPS")

            # if step == 1000:
            #     done = True
    except Exception as e:
        cam.close()
        print(e)
```

后续改进：
1. 明确帧率是否对执行效果有影响
2. 列出在验证时可以调整的模型参数
3. 明确select_action的输入输出与可调整参数
4. 列出dp模型中可以调整的参数
5. 执行场景与采集场景的差异等数据集问题排查


#### 调试日志 (2025-03-03)

需要测试上机推理的耗时

可视化当前数据集命令：
```bash
python -m lerobot.scripts.visualize_dataset --config-path /home/fourier/data/final/fourier_pnp_coke/config.yaml --output-dir /home/fourier/data/final/fourier_pnp_coke/visualize
```

确认验证时模型的参数设置


#### 调试日志 (2025-03-05)

添加了使用训练数据测试现有模型的功能，测试对数据的拟合程度，对状态数据进行可视化对比

增大bs与步数训练

预计新增padding图像功能，用于去除背景干扰

进一步可视化数据增强，确认其效果

