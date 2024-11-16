"""
Contains utilities to process raw data format of fourier dataset
"""

import gc
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

def encode_video(image_dir, video_dir, episode_id, camera, encoding, fps):
    # encode the video
    print("Encoding videos...")
    image_timestamps = []
    image_paths = sorted(list(image_dir.glob("*.png")))
    tmp_dir = video_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, image_path in enumerate(tqdm.tqdm(image_paths)):
        # save to a temporary directory with frame_00000x.png format
        tmp_path = tmp_dir / f"frame_{i:06d}.png"
        shutil.copy(image_path, tmp_path)
        image_timestamps.append(i * 1 / fps)

    encode_video_frames(tmp_dir, video_dir / f"episode_{episode_id}" / f"{camera}.mp4", fps, "libx264", overwrite=True)
    shutil.rmtree(tmp_dir)
    return f"episode_{episode_id}/{camera}.mp4", image_timestamps

def get_imgs_array(image_dir, fps):
    # get the images array
    imgs_array = []
    image_paths = sorted(list(image_dir.glob("*.png")))
    image_timestamps = []
    for i, image_path in enumerate(image_paths):
        img = PILImage.open(image_path)
        imgs_array.append(np.array(img))
        image_timestamps.append(i * 1 / fps)
    return imgs_array, image_timestamps

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
            print("Duplicate timestamp found: ", t, " trying to use next closest timestamp")
            if idx + 1 not in already_matched:
                closest_indices.append(idx + 1)
                already_matched.add(idx + 1)

    # print("closest_indices: ", len(closest_indices))
    return np.array(closest_indices)

def get_cameras(hdf5_path):
    # get camera keys
    image_path = hdf5_path.with_suffix("")
    # get all folder names in the image path
    camera_keys = [x.name for x in image_path.iterdir() if x.is_dir()]
    return camera_keys

def check_format(raw_dir) -> bool:
    # check data format
    hdf5_paths = list(raw_dir.glob("episode_*.hdf5"))
    assert len(hdf5_paths) != 0
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as data:
            assert "/action" in data
            assert "/state/robot" in data
            assert "/state/pose" in data
            assert "/state/hand" in data
            
            assert data["/action/robot"].ndim == 2
            assert data["/action/pose"].ndim == 2
            assert data["/action/hand"].ndim == 2
            
            assert data["/state/robot"].ndim == 2
            assert data["/state/pose"].ndim == 2
            assert data["/state/hand"].ndim == 2

            num_action_frames = data["/action/robot"].shape[0]
            assert num_action_frames == data["/action/pose"].shape[0]
            assert num_action_frames == data["/action/hand"].shape[0]

            num_state_frames = data["/state/robot"].shape[0]
            assert num_state_frames == data["/state/pose"].shape[0]
            assert num_state_frames == data["/state/hand"].shape[0]
                
def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
    qpos: bool = True,
):
    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    num_episodes = len(hdf5_files)

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx in tqdm.tqdm(ep_ids):
        ep_path = hdf5_files[ep_idx]
        with h5py.File(ep_path, "r") as ep:
            state = torch.from_numpy(ep["/state/robot"][:])
            if qpos:
                # concatenate the robot state with the hand state
                action = torch.from_numpy(np.concatenate([ep["/action/robot"][:], ep["/action/hand"][:]], axis=1))
            else:
                # concatenate the ee pose state with the hand state
                action = torch.from_numpy(np.concatenate([ep["/action/pose"][:], ep["/action/hand"][:]], axis=1))

            ep_dict = {}
            matched = None
            num_frames = None

            for camera in get_cameras(ep_path):
                img_key = f"observation.images.{camera}"
                image_dir = ep_path.with_suffix("") / camera
                if video:
                    fname, image_timestamps = encode_video(image_dir, videos_dir, ep_idx, camera, encoding, fps)
                    if matched is None:
                        # match the robot state timestamps with the image timestamps
                        # frequency of the robot state is higher than the image frequency
                        data_ts = np.asarray(ep["/timestamp"])
                        non_duplicate = np.where(np.diff(data_ts) > 0)[0]
                        image_ts = np.asarray([ts + data_ts[0] for ts in image_timestamps])
                        image_ts = filter(lambda x: x < data_ts[-1], image_ts)
                        matched = match_timestamps(data_ts[non_duplicate], image_ts)
                        num_frames = len(matched)
                    # store the reference to the video frame
                    ep_dict[img_key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                    ]
                else:
                    imgs_array, image_timestamps = get_imgs_array(image_dir, fps)
                    ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]
                    if matched is None:
                        # match the robot state timestamps with the image timestamps
                        # frequency of the robot state is higher than the image frequency
                        data_ts = np.asarray(ep["/timestamp"])
                        non_duplicate = np.where(np.diff(data_ts) > 0)[0]
                        image_ts = np.asarray([ts + data_ts[0] for ts in image_timestamps])
                        image_ts = filter(lambda x: x < data_ts[-1], image_ts)
                        matched = match_timestamps(data_ts[non_duplicate], image_ts)
                        num_frames = len(matched)

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True
            ep_dict["observation.state"] = state[matched]
            ep_dict["action"] = action[matched]
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
            ep_dict["next.done"] = done

            assert isinstance(ep_idx, int)
            ep_dicts.append(ep_dict)

        gc.collect()

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}
    keys = [key for key in data_dict if "observation.images." in key]
    
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
    qpos: bool = True, # Whether use qpos or ee pose for training
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 50

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding, qpos)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info