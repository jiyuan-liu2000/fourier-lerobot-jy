import os
import cv2
import h5py
import shutil
import argparse

import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count


def maintain_aspect_ratio_resize(image, target_size):
    """
    调整图像大小，保持宽高比，必要时进行裁剪
    """
    target_h, target_w = target_size[1], target_size[0]
    height, width = image.shape[:2]
    
    # 计算源图像和目标尺寸的宽高比
    aspect_ratio_orig = width / height
    aspect_ratio_target = target_w / target_h
    
    if aspect_ratio_orig > aspect_ratio_target:
        # 图像太宽，需要在宽度方向裁剪
        new_width = int(height * aspect_ratio_target)
        start_x = (width - new_width) // 2
        image = image[:, start_x:start_x+new_width]
    elif aspect_ratio_orig < aspect_ratio_target:
        # 图像太高，需要在高度方向裁剪
        new_height = int(width / aspect_ratio_target)
        start_y = (height - new_height) // 2
        image = image[start_y:start_y+new_height]
    
    # 调整到目标大小
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

def process_single_video(i, video_dir, output_dir, actions, states, timestamps, error_eps, target_size=(400, 240)):
    """处理单个视频文件并保存到HDF5。"""
    key = f'{i:09d}'
    if key in error_eps:
        return f"跳过 {key}（错误集中的视频）"
        
    video_filename = f'{key}.mp4'
    video_path = os.path.join(video_dir, video_filename)
    
    if not os.path.exists(video_path):
        return f"文件不存在：{video_path}"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"无法打开文件：{video_path}"

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 使用保持宽高比的调整函数
        resized_frame = maintain_aspect_ratio_resize(frame, target_size)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        transposed_frame = np.transpose(resized_frame, (2, 0, 1))
        frames.append(transposed_frame)

    cap.release()

    video_array = np.array(frames)
    for idx, img_array in enumerate(video_array):
        img_array = np.transpose(img_array, (1,2,0))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        video_array[idx] = np.transpose(img_array, (2,0,1))
    
    length = np.min([video_array.shape[0], actions[i].shape[0], states[i].shape[0]])
    
    hdf5_filename = f'episode_{i:09d}.hdf5'
    hdf5_path = os.path.join(output_dir, hdf5_filename)
    
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('observation.image.left', data=video_array[:length], compression="gzip")
        f.create_dataset('qpos_action', data=actions[i][:length], compression="gzip")
        f.create_dataset('observation.state', data=states[i][:length], compression="gzip")
        f.create_dataset('timestamp', data=timestamps[i][:length], compression="gzip")
    
    tqdm.write(f"处理完成：{video_filename} -> {hdf5_filename}，形状：{video_array.shape}，长度：{length}")
    return f"处理完成：{video_filename} -> {hdf5_filename}，形状：{video_array.shape}，长度：{length}"

def read_hdf5_data(file_path, start_key, end_key):
    """从HDF5文件中读取指定范围的数据。"""
    actions = []
    states = []
    timestamps = []
    error_eps = []

    with h5py.File(file_path, 'r') as f:
        print("可用的键值：", list(f.keys()))
        
        for i in tqdm(range(start_key, end_key + 1), desc="读取HDF5数据"):
            key = f'{i:09d}'
            
            if key in f:
                try:
                    action_hand = f[f'{key}/action/hand'][()]
                    action_robot = f[f'{key}/action/robot'][()]
                    state_hand = f[f'{key}/state/hand'][()]
                    state_robot = f[f'{key}/state/robot'][()]
                    timestamp = f[f'{key}/timestamp'][()]
                    
                    assert action_hand.shape[1] == 12
                    assert action_robot.shape[1] == 32
                    assert state_hand.shape[1] == 12
                    assert state_robot.shape[1] == 32
                    
                    action = np.concatenate([
                        action_hand, action_robot[:, -14:]
                    ], axis=1)
                    state = np.concatenate([
                        state_hand, state_robot[:, -14:]
                    ], axis=1)
                    
                    actions.append(action)
                    states.append(state)
                    timestamps.append(timestamp)
                except Exception as e:
                    tqdm.write(f"处理键值 {key} 时出错：{str(e)}")
                    error_eps.append(key)
                    if len(actions) > 0:
                        actions.append(action)
                        states.append(state)
                        timestamps.append(timestamp)
            else:
                print(f"HDF5文件中未找到键值 {key}")
                
    return actions, states, timestamps, error_eps

def main():
    parser = argparse.ArgumentParser(description='并行处理视频数据')
    parser.add_argument('--hdf5_path', type=str, default='/mnt/sda/diamond/df_samples/trainable_data.hdf5',
                      help='输入HDF5文件路径')
    parser.add_argument('--video_dir', type=str, default='/mnt/sda/diamond/df_samples',
                      help='输入视频目录')
    parser.add_argument('--output_dir', type=str, default='/mnt/sda/act/df_samples_processed_rgb',
                      help='输出目录')
    parser.add_argument('--target_width', type=int, default=200, help='目标图像宽度')
    parser.add_argument('--target_height', type=int, default=200, help='目标图像高度')
    parser.add_argument('--start_key', type=int, default=0, help='起始键值')
    parser.add_argument('--end_key', type=int, default=500, help='结束键值')
    parser.add_argument('--num_processes', type=int, default=16, 
                      help='使用的CPU核心数量（默认为CPU核心数-1）')

    args = parser.parse_args()
    
    # 设置进程数
    if args.num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    else:
        num_processes = min(max(1, args.num_processes), cpu_count())
    
    target_size = (args.target_width, args.target_height)
    
    print(f"使用 {num_processes} 个进程进行处理")
    print(f"目标图像尺寸：{target_size}")

    # 读取HDF5数据
    print("开始读取HDF5数据...")
    actions, states, timestamps, error_eps = read_hdf5_data(
        args.hdf5_path, args.start_key, args.end_key
    )
    print(f"已读取 {len(actions)} 个片段")

    # 准备输出目录
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"已删除现有目录 {args.output_dir}")
    os.makedirs(args.output_dir)
    print(f"已创建新目录 {args.output_dir}")

    process_func = partial(
        process_single_video,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        actions=actions,
        states=states,
        timestamps=timestamps,
        error_eps=error_eps,
        target_size=target_size
    )

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, range(args.start_key, args.end_key + 1)),
            total=args.end_key - args.start_key + 1,
            desc="processing data"
        ))

    # 打印处理结果
    # print("\n处理结果：")
    # for result in results:
    #     print(result)

if __name__ == '__main__':
    main()