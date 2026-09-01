# ParallelAlignerTemporal Data Flow Documentation

## Overview

`ParallelAlignerTemporal` extends `ParallelAligner` to handle multi-timestamp optimization for dynamic scenes. It treats T timestamps with N charts each as T×N total charts, using a multi-head MLP where n_heads=N (shared across timestamps for the same view).

## Key Concepts

### Data Dimensions
- **T**: Number of timestamps
- **N**: Number of charts per timestamp (views/cameras)
- **H×W**: Height × Width of depth maps
- **T×N**: Total number of charts across all timestamps

### Multi-Head MLP Architecture
- The deformation MLP has **N heads** (one per view/camera)
- Each head processes all timestamps for its corresponding view
- Temporal information is injected via time features concatenated to spatial features

---

## Class Initialization

### Input Data Flow

```
__init__(depths, cameras, n_timestamps, n_charts_per_timestamp, ...)
    │
    ├─> depths: (T×N, H, W)
    ├─> cameras: CamerasWrapper with T×N cameras
    ├─> timestamp_indices: (T×N,) - which timestamp each chart belongs to
    │
    └─> super().__init__()  # Initialize parent ParallelAligner
         │
         ├─> Creates spatial encodings (charts_encoding)
         ├─> Creates depth encodings (depth_encoding)
         └─> Creates confidence maps (_confidence)
```

### Component Creation

```
_create_temporal_mlp()
    │
    └─> DeformationMultiMLPTemporal(
         n_heads=N,  # Key: N heads, not T×N
         time_feature_dim=temporal_encoding_dim,
         ...
        )

TemporalEncoding(n_timestamps=T, ...)
    │
    └─> Learnable or positional encoding for T timestamps
```

**Key Point**: MLP has N heads (views), not T×N. The same view across different timestamps shares MLP parameters.

---

## Main Data Structures

### Stored Tensors

| Tensor | Shape | Device | Description |
|--------|-------|--------|-------------|
| `self._depths` | (T×N, H, W) | CPU/GPU | Original depth maps |
| `self._pts_uv` | (T×N, H*W, 2) | CPU | UV coordinates for each chart |
| `self._rays` | (T×N, H*W, 3) | CPU | Ray directions |
| `self._verts` | (T×N, H*W, 3) | CPU | 3D vertices (undeformed) |
| `self.timestamp_indices` | (T×N,) | GPU (buffer) | Timestamp index for each chart |
| `self._confidence` | (T×N, H, W) | GPU | Learnable confidence weights |

### Learned Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `charts_encoding.encodings` | (T×N, encoding_dim) | Per-chart spatial encodings |
| `depth_encoding.encodings` | (T×N, encoding_dim) | Per-chart depth encodings |
| `temporal_encoding.embeddings` | (T, time_dim) | Per-timestamp temporal encodings |
| `deformation` MLP weights | N heads | Multi-head MLP for deformations |

---

## Core Methods Data Flow

### 1. `verts_deformations` Property

Computes deformations for all T×N charts. **Called every iteration in training loop.**

```
verts_deformations (Property)
    │
    ├─> Step 1: Get Spatial Encodings
    │   pts_uv (T×N, H*W, 2) --GPU--> charts_encoding
    │   └─> encodings (T×N, H*W, encoding_dim) --CPU--
    │
    ├─> Step 2: Apply Confidence Weighting (if enabled)
    │   confidence (T×N, H, W) × encodings
    │
    ├─> Step 3: Add Depth Encodings (if enabled)
    │   depth_coords --GPU--> depth_encoding
    │   └─> depth_encodings (T×N, H*W, encoding_dim) --CPU--
    │   │
    │   └─> encodings = encodings + depth_encodings  (or multiply/replace/adaln)
    │
    ├─> Step 4: Get Temporal Features
    │   timestamp_indices (T×N,) --GPU--> temporal_encoding
    │   └─> time_features (T×N, time_dim) --CPU--
    │   └─> expand to (T×N, H*W, time_dim)
    │
    ├─> Step 5: Reorganize by View
    │   Group charts by view_idx:
    │   For view_idx in 0..N-1:
    │       charts = [t0_view_idx, t1_view_idx, ..., tT-1_view_idx]
    │       encodings_view = stack(encodings[charts])  # (T, H*W, enc_dim)
    │       time_features_view = stack(time_features[charts])  # (T, H*W, time_dim)
    │
    │   └─> encodings_mlp: (N, T*H*W, encoding_dim) --CPU--
    │   └─> time_features_mlp: (N, T*H*W, time_dim) --CPU--
    │
    ├─> Step 6: Forward Through MLP (Per-Timestamp for Memory Efficiency)
    │   Reshape to (N, T, H*W, dim)
    │
    │   For t in 0..T-1:
    │       encodings_t = encodings_mlp[:, t, :, :]  # (N, H*W, enc_dim) --CPU--
    │       time_features_t = time_features_mlp[:, t, :, :]  # (N, H*W, time_dim) --CPU--
    │
    │       --GPU--> deformation(encodings_t, time_features_t)
    │       └─> deformations_t (N, H*W, output_dim) --CPU--
    │
    │   Stack all timestamps: deformations (N, T*H*W, output_dim) --CPU--
    │
    ├─> Step 7: Reorganize to (T×N, H*W, output_dim)
    │   Permute and reshape to original chart order
    │
    ├─> Step 8: Handle Ray-Aligned Deformations
    │   if predict_along_rays:
    │       deformations = deformations * normalize(rays)
    │
    │   └─> deformations (T×N, H*W, 3) --GPU--
    │
    └─> Return: deformations --GPU--
```

**Memory Optimization**: Uses CPU offloading - processes one timestamp at a time through MLP, keeping results on CPU until final step.

---

### 2. `compute_verts_for_charts(chart_indices)`

Computes deformations for a **subset** of charts (used in memory-efficient batching).

```
compute_verts_for_charts(chart_indices)
    │
    │ Input: chart_indices (batch_size,) --GPU--
    │        Contains indices like [t0_v0, t0_v1, ..., t0_v7, t2_v0, ..., t2_v7]
    │
    ├─> Step 1: Get Encodings for Subset
    │   all_encodings (T×N, H*W, enc_dim) --GPU--
    │   └─> encodings = all_encodings[chart_indices]  # (batch, H*W, enc_dim)
    │
    ├─> Step 2: Apply Confidence & Depth Encodings
    │   (Same as verts_deformations, but only for subset)
    │
    ├─> Step 3: Get Temporal Features for Subset
    │   time_features = temporal_encoding(timestamp_indices[chart_indices])
    │   └─> (batch, H*W, time_dim) --GPU--
    │
    ├─> Step 4: Group by Timestamp
    │   For each unique timestamp t in chart_indices:
    │       positions = [i where chart_indices[i] belongs to timestamp t]
    │
    │       encodings_t = stack([encodings[i] for i in positions])  # (N, H*W, enc_dim)
    │       time_features_t = stack([time_features[i] for i in positions])  # (N, H*W, time_dim)
    │
    │       --GPU--> deformation(encodings_t, time_features_t)
    │       └─> deformations_t (N, H*W, output_dim) --GPU--
    │
    │       Store in deformations_list[positions]
    │
    ├─> Step 5: Stack and Apply Ray Alignment
    │   deformations = stack(deformations_list)  # (batch, H*W, output_dim)
    │   deformations = apply_ray_alignment(deformations, rays[chart_indices])
    │
    ├─> Step 6: Compute Deformed Vertices
    │   verts = self._verts[chart_indices]  # (batch, H*W, 3)
    │   deformed_verts = verts + deformations
    │
    └─> Return: deformed_verts (batch, H*W, 3) --GPU--
```

**Key Difference from verts_deformations**:
- Only processes **sampled timestamps**, not all T timestamps
- Each timestamp still processes all N views together (proper multi-head usage)
- Keeps data on GPU throughout (no CPU offloading)

---

### 3. `loss(reference_depths, pred_depths, masks, chart_indices)`

Override of parent loss to support batched charts.

```
loss(reference_depths, pred_depths, masks, chart_indices)
    │
    ├─> if chart_indices is not None (memory-efficient mode):
    │   │
    │   ├─> For point cloud reference:
    │   │   For each chart in chart_indices:
    │   │       actual_idx = chart_indices[i]
    │   │       reference_pts = self.reference_pts[actual_idx]
    │   │       pred_depth_i = pred_depths[i]
    │   │       camera = self.cameras.p3d_cameras[actual_idx]
    │   │
    │   │       projected_depth = get_points_depth_in_depthmap(
    │   │           pts=reference_pts,
    │   │           depthmap=pred_depth_i,
    │   │           camera=camera
    │   │       )
    │   │
    │   └─> For depth map reference:
    │       diff = pred_depths - reference_depths  # Broadcasted
    │
    ├─> Compute absolute difference
    │   diff = |diff|
    │
    ├─> Apply confidence weighting (if enabled)
    │   if chart_indices is not None:
    │       confidence = self.confidence[chart_indices]  # Only subset!
    │   else:
    │       confidence = self.confidence  # All charts
    │
    │   diff = confidence * diff - λ * log(confidence)
    │
    ├─> Apply masks (if provided)
    │   diff = masks * diff
    │
    └─> Return: diff.mean()
```

**Key Feature**: Correctly handles confidence indexing when using chart_indices.

---

## Training Loop (`optimize` Method)

### Initialization Phase

```
optimize(reference_data, masks, n_iterations, ...)
    │
    ├─> Setup
    │   ├─> enable_memory_efficient = True/False
    │   ├─> batch_timestamps = 4 (or None for full batch)
    │   └─> Print memory-efficient mode info
    │
    ├─> Preprocess Reference Data
    │   if using_pts_as_reference:
    │       reference_pts (T×N, num_points, 3)
    │       └─> reference_depths = project_to_camera_space(reference_pts)
    │   else:
    │       reference_depths (T×N, H, W)
    │
    ├─> Create Per-Timestamp Matchers (if use_matching_loss)
    │   For t in 0..T-1:
    │       chart_indices_t = [t*N, t*N+1, ..., t*N+N-1]
    │       cameras_t = CamerasWrapper([cameras[i] for i in chart_indices_t])
    │       matcher_t = Matcher3D(cameras_t, reference_depths[chart_indices_t])
    │
    ├─> Setup Optimizer
    │   prepare_for_optimization(
    │       encodings_lr, mlp_lr, confidence_lr, ...
    │   )
    │
    └─> Precompute Initial Normals/Curvatures (if needed)
        _normals = depth2normal_parallel(self._depths, self.cameras)
        _curvatures = normal2curv_parallel(_normals, ...)
```

### Main Training Loop

```
For i_iter in 0..n_iterations-1:
    │
    ├─> [STEP 1] Sample Timestamps (Memory-Efficient Mode)
    │   if enable_memory_efficient and batch_timestamps < T:
    │       sampled_timestamps = random.sample(range(T), batch_timestamps)
    │
    │       chart_indices = []
    │       for t in sampled_timestamps:
    │           chart_indices.extend(range(t*N, (t+1)*N))
    │
    │       chart_indices = tensor(chart_indices)  # (batch_timestamps*N,)
    │   else:
    │       chart_indices = None  # Use all charts
    │
    ├─> [STEP 2] Compute Deformed Vertices
    │   if chart_indices is not None:
    │       _deformed_verts_batch = compute_verts_for_charts(chart_indices)
    │   else:
    │       _deformed_verts_batch = self.verts  # All charts
    │
    ├─> [STEP 3] Compute Deformed Depths
    │   For each timestamp t in sampled (or all) timestamps:
    │       verts_t = _deformed_verts_batch[start:end]  # (N, H*W, 3)
    │       cameras_t = [cameras.p3d_cameras[i] for i in range(t*N, (t+1)*N)]
    │
    │       depths_t = cat([
    │           cam.get_world_to_view_transform().transform_points(verts_t[i])[..., 2]
    │           for i, cam in enumerate(cameras_t)
    │       ])
    │
    │       _deformed_depths_batch.append(depths_t)
    │
    │   _deformed_depths_batch: (batch_timestamps*N, H, W) --GPU--
    │
    ├─> [STEP 4] Compute Loss
    │   loss = self.loss(
    │       reference_depths=reference_depths_batch,
    │       pred_depths=_deformed_depths_batch,
    │       masks=masks_batch,
    │       chart_indices=chart_indices  # Key: pass indices for confidence
    │   )
    │
    ├─> [STEP 5] Compute Regularization Losses
    │   │
    │   ├─> Gradient Loss (if enabled)
    │   │   grad_loss = |grad(_deformed_depths) - grad(original_depths)|
    │   │
    │   ├─> Hessian Loss (if enabled)
    │   │   hess_loss = |hessian(_deformed_depths) - hessian(original_depths)|
    │   │
    │   ├─> Normal Loss (if enabled)
    │   │   if memory_efficient:
    │   │       # Extract transforms for batch only
    │   │       world_view_transforms = stack([
    │   │           cameras.gs_cameras[i].world_view_transform
    │   │           for i in chart_indices
    │   │       ])
    │   │       full_proj_transforms = stack([...])
    │   │
    │   │       _normals_batch = depth2normal_parallel(
    │   │           depths[chart_indices],
    │   │           world_view_transforms=world_view_transforms,
    │   │           full_proj_transforms=full_proj_transforms
    │   │       )
    │   │       _deformed_normals = depth2normal_parallel(
    │   │           _deformed_depths_batch,
    │   │           world_view_transforms=world_view_transforms,
    │   │           full_proj_transforms=full_proj_transforms
    │   │       )
    │   │
    │   │   normal_loss = (1 - dot(_normals_batch, _deformed_normals)).mean()
    │   │
    │   ├─> Curvature Loss (if enabled)
    │   │   _deformed_curvatures = normal2curv_parallel(_deformed_normals)
    │   │   curv_loss = |_curvatures_batch - _deformed_curvatures|.mean()
    │   │
    │   ├─> Matching Loss (if enabled, not supported in memory-efficient mode)
    │   │   For each timestamp t:
    │   │       errors, mask = matchers[t].compute_reprojection_errors(depths_t)
    │   │       matching_loss += errors.mean()
    │   │
    │   ├─> Reprojection Loss (if enabled)
    │   │   min_proj_diff = get_minimal_projections_diffs(
    │   │       points3d=_deformed_verts_batch,
    │   │       cameras=cameras,
    │   │       match_to_img=match_to_img,
    │   │       match_to_pix=match_to_pix
    │   │   )
    │   │   reproj_loss = min_proj_diff.mean()
    │   │
    │   ├─> Encodings Norm Regularization (if enabled)
    │   │   ce_norm_loss = charts_encoding(pts_uv[chart_indices]).norm().mean()
    │   │
    │   └─> Total Variation on Depth Encodings (if enabled)
    │       tv_loss = |depth_encoding[..., 1:] - depth_encoding[..., :-1]|.mean()
    │
    ├─> [STEP 6] Backpropagation
    │   loss.backward()
    │   optimizer.step()
    │   optimizer.zero_grad()
    │
    ├─> [STEP 7] Update Matchings (if specified iterations)
    │   if i_iter in matching_update_iters:
    │       For each timestamp t:
    │           deformed_depths_t = compute_all_depths[t*N:(t+1)*N]
    │           matchers[t].update_references(deformed_depths_t)
    │           matchers[t].match(matching_thr)
    │
    ├─> [STEP 8] Logging
    │   if i_iter % log_interval == 0:
    │       train_losses.append(loss.item())
    │       progress_bar.update(...)
    │
    ├─> [STEP 9] Cleanup
    │   del loss, _deformed_verts_batch, _deformed_depths_batch
    │
    │   # Only clear GPU cache occasionally (every 50 iters)
    │   if i_iter % 50 == 0:
    │       torch.cuda.empty_cache()
    │
    └─> Next iteration
```

---

## Memory-Efficient Batching Strategy

### Full Mode vs. Memory-Efficient Mode

| Mode | Charts Processed | Memory Usage | Speed |
|------|------------------|--------------|-------|
| **Full** | All T×N charts | High (all depths in GPU) | Faster |
| **Memory-Efficient** | batch_timestamps × N | Lower (~75% savings for batch=4/T=16) | Slightly slower |

### Memory-Efficient Mode Details

```
Configuration:
  batch_timestamps = 4  # Process 4 timestamps per iteration
  n_timestamps = 16     # Total timestamps
  n_charts_per_timestamp = 8  # 8 views per timestamp

Each Iteration:
  1. Sample: 4 random timestamps (e.g., [0, 5, 11, 14])
  2. Charts: 4 × 8 = 32 charts (instead of 16 × 8 = 128)
  3. Compute: Only 32 deformed vertices and depths
  4. Loss: Only on 32 charts (confidence correctly indexed)

Over Training:
  - Each timestamp sampled equally often (on average)
  - All parameters updated every iteration
  - Convergence: Similar to full mode, slightly more iterations
```

### Why It Works

1. **Multi-Head MLP Structure**: When we sample timestamps [0, 5, 11, 14], we still process all N views together for each timestamp, so the multi-head structure is properly utilized.

2. **Temporal Encoding**: Each chart gets its correct temporal encoding based on `timestamp_indices`.

3. **Gradient Flow**: All learnable parameters (encodings, MLP, confidence) receive gradients every iteration, just from a subset of charts.

4. **Stochastic Training**: Random sampling acts as additional regularization, similar to mini-batch SGD.

---

## Key Optimization Points

### 1. **CPU Offloading in `verts_deformations`**
- Encodings computed on GPU, moved to CPU
- MLP processes one timestamp at a time
- Results kept on CPU until final combination
- **Trade-off**: Slower but uses less GPU memory

### 2. **GPU-Only in `compute_verts_for_charts`**
- Everything stays on GPU
- Processes only sampled charts
- **Trade-off**: Faster but requires enough GPU memory for batch

### 3. **Minimal Cache Clearing**
- Only call `torch.cuda.empty_cache()` every 50 iterations
- Avoids expensive GPU synchronization
- **Result**: ~2-3x speedup in training

### 4. **Temporal Grouping**
- Charts always processed by complete timestamps
- All N views of a timestamp go through MLP together
- **Result**: Proper multi-head utilization, no zero-padding

---

## Common Shapes Reference

| Variable | Shape | Description |
|----------|-------|-------------|
| `depths` | (T×N, H, W) | Input depth maps |
| `cameras` | T×N cameras | Camera parameters |
| `timestamp_indices` | (T×N,) | [0,0,...,0, 1,1,...,1, ..., T-1,...,T-1] |
| `encodings` (spatial) | (T×N, H*W, encoding_dim) | Per-chart spatial features |
| `time_features` | (T×N, H*W, time_dim) | Per-chart temporal features |
| `deformations` (MLP input) | (N, batch*H*W, encoding_dim) | Organized by view |
| `deformations` (MLP output) | (N, batch*H*W, 3) | Organized by view |
| `deformations` (final) | (T×N, H*W, 3) | Organized by chart order |
| `verts` | (T×N, H*W, 3) | Deformed 3D vertices |

---

## Debugging Tips

### 1. Check Temporal Alignment
```python
# Verify timestamp_indices is correct
print(self.timestamp_indices)  # Should be [0,0,..,0, 1,1,..,1, ..., T-1,..,T-1]
assert len(self.timestamp_indices) == self.n_timestamps * self.n_charts_per_timestamp
```

### 2. Verify Chart Indexing
```python
# In memory-efficient mode
chart_indices = ...  # Sampled indices
for idx in chart_indices:
    t = idx // self.n_charts_per_timestamp
    v = idx % self.n_charts_per_timestamp
    print(f"Chart {idx}: timestamp={t}, view={v}")
```

### 3. Monitor Memory Usage
```python
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 4. Check Loss Computation
```python
# Verify confidence shapes match
if chart_indices is not None:
    assert self.confidence[chart_indices].shape[0] == pred_depths.shape[0]
```

---

## Performance Characteristics

### Typical Numbers (Example: T=16, N=8, H=288, W=512)

| Configuration | Charts/Iter | GPU Memory | Iter/sec | Convergence |
|---------------|-------------|------------|----------|-------------|
| Full batch | 128 | ~24 GB | 0.5 | 1000 iters |
| Batch=8 timestamps | 64 | ~14 GB | 0.8 | 1200 iters |
| Batch=4 timestamps | 32 | ~8 GB | 1.2 | 1500 iters |
| Batch=2 timestamps | 16 | ~5 GB | 1.5 | 2000 iters |

**Rule of thumb**: `batch_timestamps = 4` provides good balance between memory and speed.

---

## Related Files

- `parallel_aligner.py` - Parent class (spatial-only alignment)
- `multi_mlp_temporal.py` - Multi-head MLP with temporal features
- `temporal_encoding.py` - Temporal embedding (learned or positional)
- `charts_alignment_temporal.py` - High-level training script

---

## Version History

- **v1.0**: Initial implementation with CPU offloading
- **v1.1**: Fixed multi-head MLP usage (removed zero-padding)
- **v1.2**: Performance optimization (removed excessive cache clearing)
- **v1.3**: Added memory-efficient batching with proper confidence indexing
