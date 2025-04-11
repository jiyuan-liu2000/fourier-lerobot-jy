"""
This scripts demonstrates how to evaluate a pretrained policy on a Fourier GR1 robot.
It also supports recording video and state data for offline debugging.
"""

import cv2
import sys
import torch
import logging
import numpy as np
import argparse
import json
import time
import os
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms as v2
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from datetime import datetime

from controller import *
from camera import *

logging.basicConfig(level=logging.INFO)


def maintain_aspect_ratio_resize(image, target_size):
    """
    Resize image while maintaining the aspect ratio, crop if necessary
    """
    target_h, target_w = target_size[1], target_size[0]
    height, width = image.shape[:2]

    # Calculate aspect ratio of original and target size
    aspect_ratio_orig = width / height
    aspect_ratio_target = target_w / target_h

    if aspect_ratio_orig > aspect_ratio_target:
        # Image is too wide, crop in the width direction
        new_width = int(height * aspect_ratio_target)
        start_x = (width - new_width) // 2
        image = image[:, start_x : start_x + new_width]
    elif aspect_ratio_orig < aspect_ratio_target:
        # Image is too tall, crop in the height direction
        new_height = int(width / aspect_ratio_target)
        start_y = (height - new_height) // 2
        image = image[start_y : start_y + new_height]

    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    resized_image = resized_image.transpose(2, 0, 1)

    return resized_image


def transform_image(img, crop_size=(200, 200)):
    # Preprocess the image for passing it to the pretrained model
    img = img / 255.0
    img_reshape = maintain_aspect_ratio_resize(img, crop_size)
    logging.debug(f"reshape img shape:{img_reshape.shape}")
    img_reshape = torch.from_numpy(img_reshape).to(torch.float32)

    patch_h = 16
    patch_w = 16
    transform = v2.Compose(
        [
            v2.CenterCrop((patch_h * 14, patch_w * 14)),
        ]
    )

    transform_img = transform(img_reshape)
    return transform_img


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


class DataRecorder:
    """
    Records video and state data for offline debugging with precise synchronization
    """
    def __init__(self, output_dir=None, fps=30):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"record_data_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.fps = fps
        self.state_data = []
        self.action_data = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # 视频设置
        self.video_writer = None
        self.last_frame_time = None
        
        # 帧索引与时间戳映射 - 用于同步
        self.frame_timestamps = {}
        
        logging.info(f"Recording data to {self.output_dir}")
    
    def record_frame(self, image=None, state=None, action=None, visualize=True):
        """
        记录一帧数据，可以是图像+状态或动作
        使用相同的时间戳和ID确保同步
        """
        current_time = time.time() - self.start_time
        
        # 如果提供了图像和状态，则记录为新帧
        if image is not None and state is not None:
            self.frame_count += 1
            frame_index = self.frame_count
            
            # 保存帧时间戳用于同步
            self.frame_timestamps[frame_index] = current_time
            
            # 记录状态数据
            state_entry = {
                "timestamp": current_time,
                "frame": frame_index,
                "state": state.tolist() if isinstance(state, np.ndarray) else state
            }
            self.state_data.append(state_entry)
            
            # 初始化视频写入器
            if self.video_writer is None:
                h, w = image.shape[:2]
                
                # 尝试使用H264编码
                try:
                    # 首先尝试标准H264编码
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                    self.video_writer = cv2.VideoWriter(
                        str(self.output_dir / "robot_video.mp4"),
                        fourcc, self.fps, (w, h)
                    )
                    
                    # 验证视频写入器是否成功初始化
                    if not self.video_writer.isOpened():
                        raise Exception("H264 codec not available")
                    
                    logging.info("Using H264 codec for video recording")
                    
                except Exception as e:
                    logging.warning(f"Failed to initialize H264 codec: {e}")
                    logging.info("Trying alternative codecs...")
                    
                    # 尝试备用编解码器
                    for codec in ['avc1', 'X264', 'XVID', 'MJPG', 'mp4v']:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            self.video_writer = cv2.VideoWriter(
                                str(self.output_dir / "robot_video.mp4"),
                                fourcc, self.fps, (w, h)
                            )
                            
                            if self.video_writer.isOpened():
                                logging.info(f"Using {codec} codec for video recording")
                                break
                        except Exception:
                            continue
                    
                    # 如果所有尝试都失败，使用默认编码
                    if not self.video_writer.isOpened():
                        logging.warning("All codec attempts failed, using default codec")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(
                            str(self.output_dir / "robot_video.mp4"),
                            fourcc, self.fps, (w, h)
                        )
            
            # 将图像转换为BGR格式（如果需要）
            if len(image.shape) == 3 and image.shape[0] == 3:  # CHW格式
                image = image.transpose(1, 2, 0)
            
            # 确保图像是uint8类型
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
            
            # 写入视频
            self.video_writer.write(image)
            
            # 可视化
            if visualize:
                vis_img = image.copy()
                # 添加帧号和时间戳
                cv2.putText(vis_img, f"Frame: {frame_index} | Time: {current_time:.3f}s", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Recording", vis_img)
                cv2.waitKey(1)
            
            self.last_frame_time = current_time
        
        # 如果提供了动作，则记录动作数据
        # 寻找最接近的状态帧进行同步
        elif action is not None:
            # 找到最接近当前时间的帧索引
            closest_frame = self.frame_count
            closest_time = self.last_frame_time or current_time
            
            # 记录动作数据，关联到相同的帧
            action_entry = {
                "timestamp": closest_time,  # 使用相同的时间戳
                "frame": closest_frame,     # 使用相同的帧索引
                "action": action.tolist() if isinstance(action, np.ndarray) else action
            }
            self.action_data.append(action_entry)
    
    def save_data(self):
        """保存所有记录的数据，关闭资源"""
        if self.video_writer:
            self.video_writer.release()
            logging.info(f"Video saved to {self.output_dir}/robot_video.mp4")
        
        # 保存同步信息
        sync_data = {
            "total_frames": self.frame_count,
            "duration": self.last_frame_time,
            "fps": self.fps,
            "frame_timestamps": self.frame_timestamps
        }
        
        with open(self.output_dir / "sync_info.json", "w") as f:
            json.dump(sync_data, f, indent=2)
        
        # 保存状态数据
        with open(self.output_dir / "state_data.json", "w") as f:
            json.dump(self.state_data, f, indent=2)
        
        # 保存动作数据
        with open(self.output_dir / "action_data.json", "w") as f:
            json.dump(self.action_data, f, indent=2)
            
        logging.info(f"Recorded {self.frame_count} frames of data to {self.output_dir}")
        
        # 创建HTML查看器
        self._create_html_viewer()
        
        # 关闭可视化窗口
        cv2.destroyAllWindows()
    
    def _create_html_viewer(self):
        """Creates a simple HTML viewer for the recorded data"""
        
        # 直接读取JSON文件并内联到HTML中
        state_data_str = "[]"
        action_data_str = "[]"
        sync_data_str = "{}"
        
        try:
            with open(self.output_dir / "state_data.json", "r") as f:
                state_data_str = f.read()
            
            with open(self.output_dir / "action_data.json", "r") as f:
                action_data_str = f.read()
                
            with open(self.output_dir / "sync_info.json", "r") as f:
                sync_data_str = f.read()
        except Exception as e:
            logging.error(f"Error reading data files: {e}")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Robot Data Viewer</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f7f7f7; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .video-container {{ flex: 2; min-width: 400px; background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .data-container {{ flex: 1; min-width: 300px; padding: 15px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                video {{ width: 100%; border-radius: 4px; }}
                .current-data {{ background-color: #f5f5f5; padding: 12px; margin-top: 15px; border-radius: 6px; }}
                .controls {{ margin: 15px 0; display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }}
                button {{ padding: 8px 15px; background-color: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                button:hover {{ background-color: #3b78e7; }}
                #currentTime {{ margin-left: 15px; font-weight: bold; }}
                h1, h3 {{ color: #333; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
                th {{ background-color: #f2f2f2; text-align: center; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .time-info {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
                .time-info span {{ font-weight: bold; }}
                .highlight {{ background-color: #ffeb3b !important; }}
                .server-instructions {{ background-color: #e1f5fe; padding: 15px; margin-bottom: 20px; border-radius: 8px; display: none; }}
                .server-instructions code {{ background-color: #263238; color: white; padding: 3px 6px; border-radius: 4px; }}
                #status {{ margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 4px; }}
                .sync-info {{ display: flex; justify-content: space-between; background-color: #e8f5e9; padding: 10px; margin: 10px 0; border-radius: 4px; }}
                .sync-badge {{ background-color: #43a047; color: white; padding: 2px 6px; border-radius: 10px; font-size: 12px; margin-left: 5px; }}
            </style>
        </head>
        <body>
            <div class="server-instructions" id="serverInstructions">
                <h2>本地服务器设置</h2>
                <p>您正在通过文件协议（file://）查看此页面。为了正确加载数据文件，请使用HTTP服务器:</p>
                <ol>
                    <li>打开终端，切换到{self.output_dir}目录</li>
                    <li>运行以下命令启动Python简易HTTP服务器：<br>
                        <code>python -m http.server 8000</code></li>
                    <li>然后在浏览器中访问: <a href="#" id="serverLink">http://localhost:8000/viewer.html</a></li>
                </ol>
            </div>

            <h1>Robot Data Playback</h1>
            <div id="syncInfo" class="sync-info">
                <div>总帧数: <span id="totalFrames">-</span></div>
                <div>总时长: <span id="totalDuration">-</span>秒</div>
                <div>FPS: <span id="recordedFps">-</span></div>
            </div>
            
            <div class="container">
                <div class="video-container">
                    <video id="robotVideo" controls>
                        <source src="robot_video.mp4" type="video/mp4">
                        您的浏览器不支持视频标签
                    </video>
                    <div class="controls">
                        <button onclick="seekFrame(-1)">-1 Frame</button>
                        <button onclick="seekFrame(1)">+1 Frame</button>
                        <button onclick="document.getElementById('robotVideo').playbackRate = 0.5;">0.5x Speed</button>
                        <button onclick="document.getElementById('robotVideo').playbackRate = 1.0;">1.0x Speed</button>
                        <button onclick="document.getElementById('robotVideo').playbackRate = 2.0;">2.0x Speed</button>
                        <span id="currentTime">Time: 0.00s</span>
                        <span id="frameInfo">Frame: 0</span>
                    </div>
                </div>
                <div class="data-container">
                    <h3>Current State Data <span id="stateSync" class="sync-badge">同步</span></h3>
                    <div id="currentState" class="current-data">No data</div>
                    <h3>Current Action Data <span id="actionSync" class="sync-badge">同步</span></h3>
                    <div id="currentAction" class="current-data">No data</div>
                </div>
            </div>
            
            <div id="status">状态: 正在加载数据...</div>
            
            <script>
                // 检查页面是否通过HTTP服务器加载
                if (window.location.protocol === 'file:') {{
                    document.getElementById('serverInstructions').style.display = 'block';
                    document.getElementById('serverLink').href = 'http://localhost:8000/viewer.html';
                    document.getElementById('status').innerHTML = '状态: <span style="color:red">请使用HTTP服务器访问此页面</span>';
                }}
                
                // 初始化数据
                let stateData = {state_data_str};
                let actionData = {action_data_str};
                let syncData = {sync_data_str};
                let currentStateIndex = -1;
                let currentActionIndex = -1;
                let frameTimestamps = syncData.frame_timestamps || {{}};
                
                // 初始化同步信息
                document.getElementById('totalFrames').textContent = syncData.total_frames || stateData.length;
                document.getElementById('totalDuration').textContent = (syncData.duration || 0).toFixed(2);
                document.getElementById('recordedFps').textContent = syncData.fps || 30;
                
                // 时间到帧的映射
                function getFrameFromTime(time) {{
                    let closestFrame = 1;
                    let minDiff = Infinity;
                    
                    for(const frame in frameTimestamps) {{
                        const diff = Math.abs(frameTimestamps[frame] - time);
                        if(diff < minDiff) {{
                            minDiff = diff;
                            closestFrame = parseInt(frame);
                        }}
                    }}
                    
                    return closestFrame;
                }}
                
                // 帧到时间的映射
                function getTimeFromFrame(frame) {{
                    return frameTimestamps[frame] || 0;
                }}
                
                // 更新数据显示函数
                function updateDataDisplay(currentTime) {{
                    document.getElementById('currentTime').textContent = 'Time: ' + currentTime.toFixed(3) + 's';
                    
                    const currentFrame = getFrameFromTime(currentTime);
                    document.getElementById('frameInfo').textContent = 'Frame: ' + currentFrame;
                    
                    // 找到最接近当前时间的状态数据
                    let closestState = null;
                    let minStateDiff = Infinity;
                    let stateIndex = -1;
                    
                    for(let i = 0; i < stateData.length; i++) {{
                        const state = stateData[i];
                        const diff = Math.abs(state.timestamp - currentTime);
                        if(diff < minStateDiff) {{
                            minStateDiff = diff;
                            closestState = state;
                            stateIndex = i;
                        }}
                    }}
                    
                    // 找到最接近当前时间的动作数据
                    let closestAction = null;
                    let minActionDiff = Infinity;
                    let actionIndex = -1;
                    
                    for(let i = 0; i < actionData.length; i++) {{
                        const action = actionData[i];
                        const diff = Math.abs(action.timestamp - currentTime);
                        if(diff < minActionDiff) {{
                            minActionDiff = diff;
                            closestAction = action;
                            actionIndex = i;
                        }}
                    }}
                    
                    // 更新高亮状态
                    currentStateIndex = stateIndex;
                    currentActionIndex = actionIndex;
                    
                    // 更新同步状态
                    document.getElementById('stateSync').style.backgroundColor = 
                        (closestState && closestState.frame === currentFrame) ? '#43a047' : '#f44336';
                    document.getElementById('actionSync').style.backgroundColor = 
                        (closestAction && closestAction.frame === currentFrame) ? '#43a047' : '#f44336';
                    
                    // 更新状态显示
                    if(closestState) {{
                        let stateHTML = `
                        <div class="time-info">
                            <span>Frame: ${{closestState.frame}}</span>
                            <span>Time: ${{closestState.timestamp.toFixed(3)}}s</span>
                        </div>`;
                        
                        // 如果状态是数组，显示为表格
                        if(Array.isArray(closestState.state)) {{
                            stateHTML += '<table><tr><th>Index</th><th>Value</th></tr>';
                            for(let i = 0; i < closestState.state.length; i++) {{
                                stateHTML += `<tr><td>${{i}}</td><td>${{formatNumber(closestState.state[i])}}</td></tr>`;
                            }}
                            stateHTML += '</table>';
                        }} else {{
                            // 否则显示为对象
                            stateHTML += `<pre>${{JSON.stringify(closestState.state, null, 2)}}</pre>`;
                        }}
                        
                        document.getElementById('currentState').innerHTML = stateHTML;
                    }}
                    
                    // 更新动作显示
                    if(closestAction) {{
                        let actionHTML = `
                        <div class="time-info">
                            <span>Frame: ${{closestAction.frame}}</span>
                            <span>Time: ${{closestAction.timestamp.toFixed(3)}}s</span>
                        </div>`;
                        
                        // 如果动作是数组，显示为表格
                        if(Array.isArray(closestAction.action)) {{
                            actionHTML += '<table><tr><th>Index</th><th>Value</th></tr>';
                            for(let i = 0; i < closestAction.action.length; i++) {{
                                actionHTML += `<tr><td>${{i}}</td><td>${{formatNumber(closestAction.action[i])}}</td></tr>`;
                            }}
                            actionHTML += '</table>';
                        }} else {{
                            // 否则显示为对象
                            actionHTML += `<pre>${{JSON.stringify(closestAction.action, null, 2)}}</pre>`;
                        }}
                        
                        document.getElementById('currentAction').innerHTML = actionHTML;
                    }}
                }}
                
                // 格式化数字，保留4位小数
                function formatNumber(num) {{
                    if (typeof num === 'number') {{
                        return num.toFixed(4);
                    }}
                    return num;
                }}
                
                // 按帧查找
                function seekFrame(offset) {{
                    const video = document.getElementById('robotVideo');
                    const currentTime = video.currentTime;
                    const currentFrame = getFrameFromTime(currentTime);
                    const targetFrame = currentFrame + offset;
                    
                    if(frameTimestamps[targetFrame]) {{
                        video.currentTime = frameTimestamps[targetFrame];
                    }} else {{
                        // 如果没有精确的帧时间戳，则按照fps估算
                        video.currentTime += offset * (1/syncData.fps);
                    }}
                }}
                
                console.log("Data loaded:", stateData.length, "state entries,", actionData.length, "action entries");
                
                // 初始化视频时间更新事件侦听器
                const videoElement = document.getElementById('robotVideo');
                videoElement.addEventListener('timeupdate', function() {{
                    updateDataDisplay(this.currentTime);
                }});
                
                // 视频加载完成时更新一次
                videoElement.addEventListener('loadeddata', function() {{
                    console.log("Video loaded, duration:", this.duration);
                    updateDataDisplay(0);
                    document.getElementById('status').innerHTML = `状态: 数据加载完成，准备就绪`;
                }});
            </script>
        </body>
        </html>
        """
        
        with open(self.output_dir / "viewer.html", "w") as f:
            f.write(html_content)
        
        logging.info(f"Created HTML viewer at {self.output_dir}/viewer.html")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate policy and record data")
    parser.add_argument("--policy_path", type=str, default="/mnt/sda/data/models/02-26-14-55_real_world_diffusion_pnp_coke_arm_loss2_horizon64/checkpoints/300000/pretrained_model",
                        help="Path to pretrained policy")
    parser.add_argument("--policy_mode", type=str, default="dp", choices=["act", "dp"],
                        help="Policy type (act or dp)")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Enable recording of video and state data")
    parser.add_argument("--record_dir", type=str, default=None,
                        help="Directory to save recorded data (default: auto-generated)")
    parser.add_argument("--vis", action="store_true", default=True,
                        help="Visualize camera feed during execution")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Maximum number of steps to run (0 for unlimited)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load the pretrained policy
    policy = load_policy(args.policy_path, policy_mode=args.policy_mode)

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

    # Initialize the GR1 player
    cam = OakCamera(rgb_resolution=(640, 480), depth_resolution=(640, 480), fps=30)
    player = GR1Player(
        OmegaConf.load("controller/configs/gr1t2_upper_body.yaml"), camera=cam
    )
    
    # Initialize data recorder if recording is enabled
    recorder = None
    if args.record:
        recorder = DataRecorder(args.record_dir, fps=30)
    
    # Move robot to the initial position
    player.reset_robot()

    step = 0
    done = False
    vis_img = args.vis
    try:
        start_time = time.time()
        while not done:
            # Get observation from the robot
            state, left_image = player.observe(mode="bimanual")
            state = torch.from_numpy(state).to(torch.float32)
            logging.debug(f"get observation img shape :{left_image.shape}")
            logging.debug(f"get observation state shape :{state.shape}")
            
            # Record raw data before any processing
            if recorder:
                # Make a copy of the original image for recording
                original_image = left_image.copy()
                recorder.record_frame(original_image, state.cpu().numpy(), visualize=args.vis)

            # Process image for model input
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
            
            # Record action data
            if recorder:
                recorder.record_frame(None, None, numpy_action)

            # Step through the environment and receive a new observation
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

            # Check if we should stop
            if args.max_steps > 0 and step >= args.max_steps:
                logging.info(f"Reached maximum number of steps ({args.max_steps})")
                done = True
                
    except KeyboardInterrupt:
        logging.info("Execution interrupted by user")
    except Exception as e:
        logging.error(f"Error during execution: {e}")
    finally:
        # Clean up
        if recorder:
            logging.info("Saving recorded data...")
            recorder.save_data()
        cam.close()
        logging.info("Camera closed")
        
        print(f"Execution completed with {step} steps")
        if recorder:
            print(f"Data saved to {recorder.output_dir}")
            print(f"You can view the data using {recorder.output_dir}/viewer.html")
