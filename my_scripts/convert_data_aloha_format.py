import os
import cv2
import time
import h5py
import shutil
import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import concurrent.futures
from multiprocessing import cpu_count
from datetime import datetime, timezone


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
def iso_to_datetime(filename: str) -> datetime:
    """
    Convert an ISO 8601 filename with microseconds back to a datetime object.
    Args:
        filename (str): The filename to parse, including or excluding the file extension.
    Returns:
        datetime: Parsed datetime object.
    """
    if "." in filename:
        base_name = filename.split(".")[0]  # Remove file extension
    else:
        base_name = filename
    return datetime.strptime(base_name, "%Y-%m-%dT%H-%M-%S_%f").replace(tzinfo=timezone.utc)
def load_frame_timestamps_from_json(json_path):
    import json
    with open(json_path, 'r') as f:
        timestamps = json.load(f)
        return np.array([iso_to_datetime(ts).timestamp() for ts in timestamps], dtype=np.float64)
def get_match_idx(frame_timestamps_path, data_ts):
    frame_ts = load_frame_timestamps_from_json(frame_timestamps_path)
    start_ts = max(frame_ts[0], data_ts[0])
    end_ts = min(frame_ts[-1], data_ts[-1])
    # print(f"Episode start from {start_ts}s to {end_ts}s")

    # discard frames before and after the timestamps
    frame_ts = [ts for ts in frame_ts if ts >= start_ts and ts <= end_ts]
    # video_ids = [id for id in video_ids if int(id) >= start_ts * 20 and int(id) <= end_ts * 20]
    data_ts = [ts for ts in data_ts if ts >= start_ts and ts <= end_ts]

    # match video and data timestamps
    matched_ts = match_timestamps(data_ts, frame_ts)

    return matched_ts

def match_timestamps(candidate, ref):
    closest_indices = []
    # candidate = np.sort(candidate)
    already_matched = set()
    for t in ref:
        idx = np.searchsorted(candidate, t, side="left")
        if idx > 0 and (idx == len(candidate) or np.fabs(t - candidate[idx - 1]) < np.fabs(t - candidate[idx])):
            idx = idx - 1
        if idx not in already_matched:
            closest_indices.append(idx)
            already_matched.add(idx)
        else:
            # print(f"Duplicate timestamp found: {t} and {candidate[idx]} trying to use next closest timestamp")
            if idx + 1 not in already_matched:
                closest_indices.append(idx + 1)
                already_matched.add(idx + 1)

    # print("closest_indices: ", len(closest_indices))
    return np.array(closest_indices)

def get_match_frame(matched_ts, frames):
    if matched_ts[0] == 0:
        del frames[0]
    if len(matched_ts) == len(frames)-1:
        del frames[-1]
    return frames
def process_single_eposide(hdf5_file, input_dir, output_dir, index, target_size=(400, 240)):
    """Process a single video file and save to HDF5."""
    # print(f"Processing {hdf5_file}")
    video_path = Path(input_dir / hdf5_file.stem / 'top/rgb').with_suffix('.mp4')
    timestamps_path = Path(input_dir / hdf5_file.stem / 'top/timestamps.json')
    if not video_path.exists():
        logging.error(f"MP4 file {video_path} does not exist, skip file {hdf5_file}.")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open file: {video_path}")
        return False

    try:
        actions, states, timestamps = read_hdf5_data(hdf5_file)
        matched_ts = get_match_idx(timestamps_path, timestamps)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize using the maintain_aspect_ratio_resize function
            resized_frame = maintain_aspect_ratio_resize(frame, target_size)
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            transposed_frame = np.transpose(resized_frame, (2, 0, 1))
            frames.append(transposed_frame)
        cap.release()
        matched_frames = get_match_frame(matched_ts, frames)
        video_array = np.array(matched_frames)
        
        
        hdf5_filename = f'episode_{index:09d}.hdf5'
        hdf5_path = os.path.join(output_dir, hdf5_filename)
        
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('observation.image.left', data=video_array, compression="gzip")
            f.create_dataset('qpos_action', data=actions[matched_ts], compression="gzip")
            f.create_dataset('observation.state', data=states[matched_ts], compression="gzip")
            f.create_dataset('timestamp', data=timestamps[matched_ts], compression="gzip")
        logging.info(f"Processed file success to {hdf5_path}")
        print(f"Processed file success to {hdf5_path}")
        return True
    except Exception as e:
        logging.exception(f"Exception processing {hdf5_file}: {e}")
        return False
def read_hdf5_data(file_path):
    """Read data from HDF5 file within the specified range of keys."""

    with h5py.File(file_path, 'r') as f:
        try:
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
        except Exception as e:
            tqdm.write(f"Error hdf5 file {file_path}")
                
    return action, state, timestamp

def main():
    parser = argparse.ArgumentParser(description='Process video data in parallel')
    parser.add_argument('--data_dir', type=str, default='/mnt/nas/DataFactory/raw/data-samples',
                      help='Input video directory')
    parser.add_argument('--output_dir', type=str, default='/mnt/sda/lerobot/samples_sim',
                      help='Output directory')
    parser.add_argument('--target_width', type=int, default=224, help='Target image width')
    parser.add_argument('--target_height', type=int, default=224, help='Target image height')
    parser.add_argument('--num_eposide', type=int, default=500, help='End key')
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
    data_dir = Path(args.data_dir)
    hdf5_files = sorted(list(data_dir.glob('*.hdf5')))
    if args.num_eposide is not None:
        hdf5_files = hdf5_files[:args.num_eposide ]
        print(f"Read {len(hdf5_files)} segments")
    if not hdf5_files:
        raise ValueError(f"No hdf5 files found in {args.data_dir}")
    

    # Prepare the output directory
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"Deleted existing directory {args.output_dir}")
    os.makedirs(args.output_dir)
    print(f"Created new directory {args.output_dir}")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        future_to_index = {
            executor.submit(process_single_eposide, hdf5_file, data_dir, args.output_dir, idx, target_size): idx
            for idx, hdf5_file in enumerate(hdf5_files, start=0)
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_index),
                           total=len(future_to_index), desc="Processing files"):
            idx = future_to_index[future]
            try:
                result = future.result()
            except Exception as exc:
                result = False
                raise ValueError(f'File at index {idx} generated an exception: {exc}')
            results[idx] = result

    num_processed = sum(1 for r in results.values() if r)

if __name__ == '__main__':
    start_time = time.time()    
    main()
    process_time = time.time() - start_time
    hours, remainder = divmod(process_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"process time: {int(hours)}hours {int(minutes)}minutes {int(seconds)}seconds")
    print(f"now time: {time.strftime('%H:%M', time.localtime())}")