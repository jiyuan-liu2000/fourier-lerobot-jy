
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

