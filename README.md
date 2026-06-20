# simple-sam2

A lightweight Python wrapper around [Meta SAM2](https://github.com/facebookresearch/sam2) that makes long-video segmentation practical.

## What this adds on top of SAM2

| Problem | What simple-sam2 does |
|---|---|
| SAM2 only accepts mask OR points+box separately | **Unified prompt API** — pass mask, points, and box together in one call |
| SAM2 loads the entire video into memory | **Batch processing** — only `batch_size` frames in memory at once |
| No way to inspect/correct mid-video | **Incremental propagation** — segment N frames, check, then continue |

---

## Installation

```bash
# 1. Install SAM2 manually first (not on PyPI)
pip install git+https://github.com/facebookresearch/sam2.git

# 2. Then install simple-sam2
pip install simple-sam2
```

> **Note:** SAM2 is a dependency which is installed from Meta's GitHub repo. You still need to **download the model weights** separately (see below).

### Download SAM2 weights

```bash
# Tiny (fastest)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Small
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# Base+
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# Large (most accurate)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

The matching config files are bundled with the `sam2` package — pass the config name directly (e.g. `"sam2.1_hiera_t.yaml"`).

---

## Quick start

```python
from simple_sam2 import SAM2Service

CONFIG  = "configs/sam2.1/sam2.1_hiera_t.yaml" # or if want to use sam2 then path is :- configs/sam2/sam2_hiera_t.yaml        
WEIGHTS = "path/to/sam2.1_hiera_tiny.pt".      # make sure .pt file matches the config
VIDEO   = "path/to/my_video.mp4"

#Note:- you only have to download weights (.pt) files, config files are already present at configs/ dir of sam2

svc = SAM2Service(cfg=CONFIG, ckpt=WEIGHTS, batch_size=60)
```

### 1. Initialise the video

```python
info = svc.init_video("demo", video_path=VIDEO)
print(f"Video has {info['total_frames']} frames")
print(f"Frames extracted to: {info['frame_dir']}")
print(f"Masks will be saved to: {info['masks_dir']}")
```

This creates the following layout under `simple_sam2_storage/` in your current directory:

```
simple_sam2_storage/
└── demo/
    ├── frames/       ← extracted JPEG frames (00000.jpg, 00001.jpg, …)
    ├── masks/        ← output masks written here
    └── tmp_batches/  ← transient working dirs (auto-managed)
```

If frames are already extracted from a previous run, extraction is skipped automatically.

### 2. Add prompts

```python
# Positive click only
svc.add_prompts("demo", frame_idx=0, obj_id=1, pos_points=[[320, 240]])

# Or mix points + bounding box
svc.add_prompts(
    "demo", frame_idx=0, obj_id=2,
    pos_points=[[400, 300]],
    neg_points=[[100, 100]],
    box=[200, 150, 600, 480],
)
```

### 3. Propagate and save

```python
n = svc.propagate_and_save(
    "demo",
    start_frame_idx=0,
    end_frame_idx=5,                       # optional, defaults to end of video
    obj_labels={"1": "person", "2": "car"},
    progress_callback=lambda p: print(f"Progress: {p:.1f}%"),
)
print(f"Done! Saved {n} masks to {info['masks_dir']}")
```

Output filenames follow the pattern `<frame_index>_<obj_id>_<label>.png`:

```
masks/
├── 00000_1_person.png
├── 00001_1_person.png
├── 00000_2_car.png
...
```

### 4. Clean up

```python
svc.clear_video("demo")                    # free GPU memory
svc.clear_video("demo", delete_storage= True)  # delete all video related frames, tmp_batches, masks
```

---

## API reference

### `SAM2Service(cfg, ckpt, batch_size=60, storage_dir=...)`

| Parameter | Description |
|---|---|
| `cfg` | SAM2 YAML config name concatenated to configs/sam2.1/ for sam2.1 or configs/sam2/ for sam2 (e.g. `"configs/sam2.1/sam2.1_hiera_t.yaml"`) |
| `ckpt` | Path to SAM2 `.pt` checkpoint file |
| `batch_size` | Frames in memory at once. Reduce if you run out of GPU memory |
| `storage_dir` | Root directory for all data. Defaults to `<cwd>/simple_sam2_storage` |

### `init_video(video_name, video_path)`

Extracts frames and sets up storage. Returns a dict with `total_frames`, `total_batches`, `batch_size`, `frame_dir`, `masks_dir`.

### `add_prompts(video_name, frame_idx, obj_id, pos_points, neg_points, box, binary_mask)`

All prompt types are optional — provide at least one. Returns a uint8 preview mask (0 or 255).

### `propagate_and_save(video_name, out_dir, start_frame_idx, end_frame_idx, obj_labels, progress_callback)`

Propagates all tracked objects and saves mask PNGs. `out_dir` defaults to `simple_sam2_storage/<video_name>/masks/`. Returns the number of files saved.

### `clear_video(video_name, delete_storag=False)`

Frees GPU memory and removes temp files. Pass `delete_storage= True` to also delete the extracted frames,masks directory.

---

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

SAM2 is © Meta Platforms, Inc. and affiliates, also under Apache 2.0.
