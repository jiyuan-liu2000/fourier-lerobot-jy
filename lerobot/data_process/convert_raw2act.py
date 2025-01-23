import os
import cv2
import time
import h5py
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count


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
        image = image[:, start_x:start_x+new_width]
    elif aspect_ratio_orig < aspect_ratio_target:
        # Image is too tall, crop in the height direction
        new_height = int(width / aspect_ratio_target)
        start_y = (height - new_height) // 2
        image = image[start_y:start_y+new_height]
    
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

def process_single_video(i, video_dir, output_dir, actions, states, timestamps, error_eps, target_size=(400, 240)):
    """Process a single video file and save to HDF5."""
    key = f'{i:09d}'
    if key in error_eps:
        tqdm.write(f"Skipping {key} (in error list)")
        return f"Skipping {key} (in error list)"
        
    video_filename = f'{key}.mp4'
    video_path = os.path.join(video_dir, video_filename)
    
    if not os.path.exists(video_path):
        tqdm.write(f"File does not exist: {video_path}")
        return f"File does not exist: {video_path}"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        tqdm.write(f"Cannot open file: {video_path}")
        return f"Cannot open file: {video_path}"

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize using the maintain_aspect_ratio_resize function
        resized_frame = maintain_aspect_ratio_resize(frame, target_size)
        # resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
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
    
    # tqdm.write(f"Processed: {video_filename} -> {hdf5_filename}, shape: {video_array.shape}, length: {length}")
    return f"Processed: {video_filename} -> {hdf5_filename}, shape: {video_array.shape}, length: {length}"

def read_hdf5_data(file_path, start_key, end_key):
    """Read data from HDF5 file within the specified range of keys."""
    actions = []
    states = []
    timestamps = []
    error_eps = []

    with h5py.File(file_path, 'r') as f:
        print("Available keys:", list(f.keys()))
        
        for i in tqdm(range(start_key, end_key + 1), desc="Reading HDF5 data"):
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
                        action_robot[:, -14:], action_hand
                    ], axis=1)
                    state = np.concatenate([
                        state_robot[:, -14:], state_hand, 
                    ], axis=1)
                    
                    actions.append(action)
                    states.append(state)
                    timestamps.append(timestamp)
                except Exception as e:
                    tqdm.write(f"Error processing key {key}: {str(e)}")
                    error_eps.append(key)
                    if len(actions) > 0:
                        actions.append(action)
                        states.append(state)
                        timestamps.append(timestamp)
            else:
                tqdm.write(f"Key {key} not found in HDF5 file.")
                
    return actions, states, timestamps, error_eps

def main():
    parser = argparse.ArgumentParser(description='Process video data in parallel')
    parser.add_argument('--hdf5_path', type=str, default='/mnt/sda/diamond/df_samples/trainable_data.hdf5',
                      help='Input HDF5 file path')
    parser.add_argument('--video_dir', type=str, default='/mnt/sda/diamond/df_samples',
                      help='Input video directory')
    parser.add_argument('--output_dir', type=str, default='/mnt/sda/act/df_samples_v2_sim',
                      help='Output directory')
    parser.add_argument('--target_width', type=int, default=200, help='Target image width')
    parser.add_argument('--target_height', type=int, default=200, help='Target image height')
    parser.add_argument('--start_key', type=int, default=0, help='Start key')
    parser.add_argument('--end_key', type=int, default=500, help='End key')
    parser.add_argument('--num_processes', type=int, default=16, 
                      help='Number of CPU cores to use (default is CPU count - 1)')

    args = parser.parse_args()
    
    # Set number of processes
    if args.num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    else:
        num_processes = min(max(1, args.num_processes), cpu_count())
    
    target_size = (args.target_width, args.target_height)
    
    print(f"Using {num_processes} processes")
    print(f"Target image size: {target_size}")

    # Read data from HDF5
    print("Reading HDF5 data...")
    actions, states, timestamps, error_eps = read_hdf5_data(
        args.hdf5_path, args.start_key, args.end_key
    )
    print(f"Read {len(actions)} segments")

    # Prepare the output directory
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"Deleted existing directory {args.output_dir}")
    os.makedirs(args.output_dir)
    print(f"Created new directory {args.output_dir}")

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
            desc="Processing data"
        ))

if __name__ == '__main__':
    start_time = time.time()    
    main()
    process_time = time.time() - start_time
    hours, remainder = divmod(process_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"process time: {int(hours)}hours {int(minutes)}minutes {int(seconds)}seconds")
    print(f"now time: {time.strftime('%H:%M', time.localtime())}")