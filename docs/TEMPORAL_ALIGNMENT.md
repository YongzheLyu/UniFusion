# Temporal Charts Alignment

Multi-timestamp extension of the charts alignment pipeline.

## Overview

This module extends the single-frame `align_charts.py` to handle multiple timestamps jointly. The key innovation is treating T timestamps with N charts each as T×N total charts, where charts from the same view across different timestamps share MLP parameters while having independent spatial encodings.

**Key Design:**
- **MLP heads**: `n_heads = N` (number of charts per timestamp)
- **Temporal features**: Added as additional input to MLP via learned embeddings or sinusoidal encoding
- **Shared across time**: Same view across timestamps shares MLP parameters
- **Independent per timestamp**: Each timestamp has its own chart/depth encodings

This design enables:
- ✅ Temporal consistency through shared MLP weights
- ✅ Per-timestamp flexibility through independent encodings
- ✅ No new loss functions needed (reuses all existing losses)
- ✅ Memory efficient (scales linearly with T)

## Data Structure

Expected directory structure:
```
/path/to/data/
├── frame_00000/
│   └── mast3r_sfm/
│       ├── images/
│       │   ├── cam00_*.jpg
│       │   ├── cam01_*.jpg
│       │   └── ...
│       ├── sparse/
│       └── ...
├── frame_00001/
│   └── mast3r_sfm/
│       └── ...
├── frame_00002/
└── ...
```

## Usage

### Basic Example

Process all frames in a directory:
```bash
python scripts/align_charts_temporal.py \
    -d /path/to/frames_output \
    -o /output/temporal_charts \
    --temporal_encoding_type learned \
    --temporal_encoding_dim 8
```

### Process Subset of Frames

Process frames 0-10 with step 2 (every other frame):
```bash
python scripts/align_charts_temporal.py \
    -d /path/to/data \
    --start_frame 0 \
    --end_frame 10 \
    --frame_step 2 \
    -o /output/temporal_charts
```

Process only first 5 frames:
```bash
python scripts/align_charts_temporal.py \
    -d /path/to/data \
    --max_frames 5 \
    -o /output/temporal_charts
```

### Parameters

**Required:**
- `-d, --data_dir`: Base directory containing `frame_XXXXX` subdirectories

**Optional:**
- `-o, --output_path`: Output directory (default: `{data_dir}/temporal_charts`)
- `-c, --config`: Config file name (default: `temporal_default`)

**Frame Selection:**
- `--start_frame`: Starting frame index (default: 0)
- `--end_frame`: Ending frame index (default: process all)
- `--frame_step`: Frame step size (default: 1, process every frame)
- `--max_frames`: Maximum number of frames to process (default: no limit)

**Temporal Encoding:**
- `--temporal_encoding_type`: Type of temporal encoding
  - `learned` (default): Learnable embeddings for each timestamp
  - `positional`: Sinusoidal positional encoding
- `--temporal_encoding_dim`: Dimension of temporal features (default: 8)

**MASt3R:**
- `--mast3r_subdir`: Subdirectory name containing MASt3R data (default: `mast3r_sfm`)

**DepthAnything:**
- `--depthanythingv2_checkpoint_dir`: Path to checkpoints (default: `./Depth-Anything-V2/checkpoints/`)
- `--depthanything_encoder`: Encoder type (default: `vitl`)

### Configuration

Edit `configs/charts_alignment/temporal_default.yaml` to customize:
- Number of iterations
- Loss weights
- Learning rates
- Regularization settings

## Architecture Details

### Data Flow

```
Input: Directory with frame_XXXXX subdirectories
├─> Auto-discover frames: [frame_00000, frame_00001, ...]
├─> Filter by: start_frame, end_frame, frame_step, max_frames
├─> For each frame: Load from {frame_dir}/mast3r_sfm/
│
├─> Combined depths: (T×N, H, W)
├─> Timestamp indices: [0,0,...,0, 1,1,...,1, ..., T-1,T-1,...,T-1]
│
├─> Spatial encodings (per timestamp): (N, H, W, encoding_dim)
├─> Temporal features: (T×N, time_dim)
│
├─> Reorganize by view: (N, T*H*W, encoding_dim + time_dim)
├─> MLP forward: (N, T*H*W, 1 or 3)
│
└─> Reorganize to: (T×N, H*W, 1 or 3) → deformations
```

### Memory Usage

For T=10 timestamps, N=8 charts, H×W=224×224:
- **Encodings**: ~10 MB (dominated by chart encodings)
- **MLP parameters**: ~100 KB (shared across time)
- **Runtime memory**: ~2 GB (for optimization)

Scales **linearly** with T (unlike naive T×N approach which scales quadratically).

## Output

For each processed timestamp t, saves:
```
{output_path}/timestamp_{t:04d}_charts_data.npz
```

Contains:
- `prior_depths`: Initial depths (N, H, W)
- `depths`: Optimized depths (N, H, W)
- `pts`: Optimized 3D points (N, H, W, 3)
- `confs`: Confidence maps (N, H, W)
- `scale_factor`: Scene scaling factor
- `timestamp`: Timestamp index

## Example Workflows

### Workflow 1: Process All Frames

```bash
# Process all 100 frames in the directory
python scripts/align_charts_temporal.py \
    -d /path/to/frames_output \
    -o /output/all_frames
```

### Workflow 2: Quick Test on First 3 Frames

```bash
# Test on first 3 frames only
python scripts/align_charts_temporal.py \
    -d /path/to/frames_output \
    --max_frames 3 \
    -o /output/test
```

### Workflow 3: Process Every 5th Frame

```bash
# Process frames 0, 5, 10, 15, ... for faster preview
python scripts/align_charts_temporal.py \
    -d /path/to/frames_output \
    --frame_step 5 \
    -o /output/keyframes
```

### Workflow 4: Process Specific Range

```bash
# Process frames 10 through 30
python scripts/align_charts_temporal.py \
    -d /path/to/frames_output \
    --start_frame 10 \
    --end_frame 30 \
    -o /output/range_10_30
```

## Advanced Usage

### Custom Temporal Encoding

Use sinusoidal temporal encoding with higher dimension:
```bash
python scripts/align_charts_temporal.py \
    -d /path/to/data \
    --temporal_encoding_type positional \
    --temporal_encoding_dim 16
```

### Custom Configuration

Create your own config file:
```bash
# Copy and modify
cp configs/charts_alignment/temporal_default.yaml configs/charts_alignment/my_config.yaml

# Use it
python scripts/align_charts_temporal.py \
    -d /path/to/data \
    -c my_config
```

### Integration with Downstream Tasks

Load temporal charts in Python:
```python
import numpy as np

# Load all timestamps
temporal_charts = []
for t in range(n_timestamps):
    data_t = np.load(f'output/timestamp_{t:04d}_charts_data.npz')
    temporal_charts.append({
        'depths': data_t['depths'],
        'pts': data_t['pts'],
        'confs': data_t['confs'],
    })

# Access specific timestamp and chart
pts_t2_chart3 = temporal_charts[2]['pts'][3]  # (H, W, 3)
```

## Files Created

```
MAtCha/
├── matcha/
│   ├── dm_deformation/
│   │   ├── multi_mlp_temporal.py          # MLP with time feature support
│   │   └── temporal_encoding.py           # Temporal encoding module
│   ├── dm_scene/
│   │   └── parallel_aligner_temporal.py   # Temporal ParallelAligner
│   └── dm_trainers/
│       └── charts_alignment_temporal.py   # Main alignment function
├── scripts/
│   └── align_charts_temporal.py           # Main script (auto-discovers frames)
├── configs/charts_alignment/
│   └── temporal_default.yaml              # Configuration
└── docs/
    └── TEMPORAL_ALIGNMENT.md              # This file
```

## Troubleshooting

**Q: "No frame_* directories found"**
- Check that your data directory contains subdirectories named `frame_00000`, `frame_00001`, etc.
- Verify the path is correct

**Q: "MASt3R scene not found"**
- Check that each frame directory has a `mast3r_sfm` subdirectory
- Use `--mast3r_subdir` if your MASt3R data is in a different subdirectory

**Q: Out of memory error**
- Use `--max_frames` to process fewer frames at once
- Use `--frame_step` to skip frames
- Lower `max_img_size` in config
- Use lower resolution pointmaps

**Q: Slow convergence**
- Increase `n_iterations` in config
- Try `--temporal_encoding_type positional`
- Adjust learning rates in config

**Q: Temporal jittering in results**
- Increase `--temporal_encoding_dim`
- Process more frames together (temporal context helps)
- Check if input frames have good temporal overlap

## Implementation Notes

**Why this design?**
1. **Shared MLP across time**: Encourages temporal consistency
2. **Independent encodings**: Allows per-frame flexibility
3. **Time as feature**: Simple yet effective for temporal awareness
4. **No new losses**: Reuses battle-tested single-frame losses
5. **Auto-discovery**: Automatically finds and processes frame directories

**Design Philosophy:**
- Simple to use: Just point to a directory
- Flexible filtering: Control which frames to process
- Scalable: Process 3 frames or 100 frames with same command
- Memory efficient: Linear scaling with number of frames

## Citation

If you use this temporal extension, please cite the original MAtCha paper along with acknowledging this extension.
