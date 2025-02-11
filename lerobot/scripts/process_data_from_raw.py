import os
import cv2
import numpy as np
import h5py

from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

key_front = 'pb'
episode_length = 1000
inter_step = 1
start_idx = 0

TASK = {
    'pp': 'pick and place lemon on the plate',
    'pb': 'pour beans into the cup'
}
task_names_key = TASK.keys()
task_counts = {}
for key in task_names_key:
    task_counts[key] = 0

def smooth_trajectory(data, method="pchip", threshold=3, sigma=1.4, plot_results=False):
    """
    平滑处理一条长度为64的轨迹，去除最大最小值，尽量穿过均值。

    参数:
    - data: 原始轨迹数据，长度为64的NumPy数组。
    - method: 平滑方法，可以选择'pchip'或'gaussian'。
    - threshold: 异常值去除的阈值，默认为3倍标准差。
    - sigma: 高斯滤波器的标准差，默认为1.5。
    - plot_results: 是否绘制结果图，默认为False。

    返回:
    - smoothed_data: 平滑处理后的轨迹数据，长度为64的NumPy数组。
    """

    def remove_outliers(data, threshold=3):
        """Remove outliers based on standard deviation."""
        mean = np.mean(data)
        std = np.std(data)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        # Create a mask for non-outliers
        mask = (data >= lower_bound) & (data <= upper_bound)

        # Replace outliers with the mean
        data_cleaned = np.where(mask, data, mean)

        return data_cleaned

    def smooth_with_pchip(data):
        x = np.arange(len(data))
        pchip = PchipInterpolator(x, data)
        y_smooth = pchip(x)
        return y_smooth

    def smooth_with_gaussian(data, sigma=1.5):
        """Smooth data using a Gaussian filter."""
        return gaussian_filter1d(data, sigma=sigma)

    # 1. 去除异常值
    data_cleaned = remove_outliers(data, threshold)

    smoothed_data = smooth_with_gaussian(data_cleaned, sigma)

    return smoothed_data

def find_hdf5_files(directory):
    """查找指定目录下的所有 .hdf5 文件"""
    hdf5_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files

def read_video_to_array(video_path, max_frames):
    """使用 OpenCV 读取 mp4 视频并存储为 NumPy 数组"""
    cap = cv2.VideoCapture(video_path)
    timestampp = []
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    frames = []
    frame_count = 0
    target_size=(400, 240)
    dec_i = 0
    decimation = 4

    while frame_count < max_frames:
        ret, frame = cap.read()
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        frames.append(resized_frame)
        timestampp.append(ts)
        frame_count += 1
    
    cap.release()
    
    if frame_count == 0:
        print(f"⚠️  视频无效或空文件: {video_path}")
        return None

    return np.array(frames), timestampp

def align_state_to_video(state, state_timestamp, video_timestamp, method="nearest"):
    """ 对 state 进行插值，使其与 video_timestamp 对齐 """
    aligned_state = np.zeros((len(video_timestamp), state.shape[1]))  # (M, feature_dim)
    for i, ts in enumerate(video_timestamp):
        closest_idx = np.abs(state_timestamp - ts).argmin()  # 找到最接近的 state 时间戳索引
        aligned_state[i] = state[closest_idx]

    return aligned_state

def sample_video_frames(video, video_timestamp, target_frames=100):
    """ 均匀抽取 video_timestamp 和 video，使长度变为 target_frames """
    total_frames = len(video)

    if total_frames <= target_frames:
        print(f"⚠️ 视频帧数 ({total_frames}) 小于等于 {target_frames}，无需抽取")
        return video, video_timestamp

    # 计算等间距索引
    sampled_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

    # 选择对应的帧和时间戳
    sampled_video = video[sampled_indices]
    sampled_timestamps = video_timestamp[sampled_indices]

    return sampled_video, sampled_timestamps

def process_hdf5_and_videos(directory):
    """遍历所有 HDF5 文件，并读取对应的 top/rgb.mp4"""
    hdf5_files = find_hdf5_files(directory)
    # video_data = {}

    for hdf5_file in hdf5_files[:200]:
        base_name = os.path.splitext(os.path.basename(hdf5_file))[0]
        video_path = os.path.join(os.path.dirname(hdf5_file), base_name, "top", "rgb.mp4")
        
        if os.path.exists(video_path):
            print(f"📂  读取视频: {video_path}")
            video_data, video_timestamp = read_video_to_array(video_path, episode_length)
            
            action = None
            state = None
            with h5py.File(hdf5_file, 'r') as f:
                action_hand = f[f'action/hand'][()]
                action_robot = f[f'/action/robot'][()]
                state_hand = f[f'state/hand'][()]
                state_robot = f[f'state/robot'][()]
                timestamp = f[f'timestamp'][()]
                
                assert action_hand.shape[1] == 12
                assert action_robot.shape[1] == 32
                assert state_hand.shape[1] == 12
                assert state_robot.shape[1] == 32
                    
                action = np.concatenate([
                    action_robot[:, -14:],
                    action_hand, 
                ], axis=1)
                state = np.concatenate([
                    state_robot[:, -14:],
                    state_hand, 
                ], axis=1)
                                
            video_timestamp += timestamp[0]
            video_data, video_timestamp = sample_video_frames(video_data, video_timestamp)
            
            
            
            state = align_state_to_video(state, timestamp, video_timestamp)
            action = align_state_to_video(action, timestamp, video_timestamp)
            
            # for i in range(action.shape[1]):
            #     action[:, i] = smooth_trajectory(action[:, i], method="gaussian", threshold=3, sigma=1.)
            #     state[:, i] = smooth_trajectory(state[:, i], method="gaussian", threshold=3, sigma=1.)
                
            video_array = []
            for frame in video_data:
                cv2.imshow('Video Playback', frame)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
                resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                transposed_frame = np.transpose(resized_frame, (2, 0, 1))
                video_array.append(transposed_frame)
                
            video_array = np.array(video_array)
            
            
            hdf5_filename = f'{TASK[key_front]}_{task_counts[key_front]}.hdf5'
            hdf5_path = os.path.join(output_dir, hdf5_filename)
            with h5py.File(hdf5_path, 'w') as f:
                # 创建一个数据集来存储图像数据
                length = episode_length
                print([video_array.shape[0], action.shape[0]], length)
                f.create_dataset('observation.image.left', data=video_array[start_idx:start_idx+length:inter_step], compression="gzip")
                f.create_dataset('qpos_action', data=action[start_idx:start_idx+length:inter_step], compression="gzip")
                f.create_dataset('observation.state', data=state[start_idx:start_idx+length:inter_step], compression="gzip")
                f.create_dataset('timestamp', data=video_timestamp[start_idx:start_idx+length:inter_step], compression="gzip")
                
            task_counts[key_front] += 1
        else:
            print(f"❌  未找到视频文件: {video_path}")
    

# 运行主程序
if __name__ == "__main__":
    directory = "/mnt/sda/diamond/data-pouring"
    out_directory = "/mnt/sda/lerobot/pouring"
    output_dir = directory + '-processed_sim'
    
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"已删除现有目录 {output_dir}")
    os.makedirs(output_dir)
    print(f"已创建新目录 {output_dir}")
            
    process_hdf5_and_videos(directory)