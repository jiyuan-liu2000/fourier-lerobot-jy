import gc
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


raw_dir = Path('/mnt/nas/DataFactory/raw/data-pouring')
hdf5_files = sorted(raw_dir.glob("*.hdf5"))
num_episodes = len(hdf5_files)

ep_dicts = []
ep_ids = range(num_episodes)
for ep_idx in tqdm.tqdm(ep_ids):
    ep_path = hdf5_files[ep_idx]
    # print(ep_path)
    video_path = Path(ep_path.stem / 'top/rgb').with_suffix('.mp4')
    timestamps_path = Path(ep_path.stem / 'top/timestamps.json')
    print(ep_path.stem)