# Convert dataset to lerobot format
```bash
python /home/len9/lerobot/lerobot/scripts/push_dataset_to_hub.py --raw-dir="/media/len9/DAQ2/2024-11-13_16-01-57" --local-dir="data/test" --video=1 --raw-format="fourier" --repo-id="lerobot/fourier" --force-override=1 --fps=50 --push-to-hub=0
```
# Visualize dataset
```bash
DATA_DIR='/home/len9/lerobot/data/test' python lerobot/scripts/visualize_dataset.py --repo-id="" --episode-index=1 --root="/home/len9/lerobot/data/test"
```
# Train ACT policy
```bash
python lerobot/scripts/train.py policy=act_fourier_real env=fourier_real
```