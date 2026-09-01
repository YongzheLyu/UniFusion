# MAtCha Camera Splitting Implementation

This document describes the camera splitting implementation for training and testing MAtCha Gaussians with 5 training cameras and 3 testing cameras from a multi-view dataset.

## Overview

The original MAtCha implementation processes all cameras together for both training and testing. This modified implementation:

1. **Splitically separates cameras** into training and testing sets (5 train + 3 test)
2. **Ensures no data leakage** between train and test sets
3. **Builds charts only from training cameras** for priors
4. **Proper validation and logging** throughout the pipeline

## Key Changes

### 1. MultipleView Dataset (`2d-gaussian-splatting/scene/multipleview_dataset.py`)

#### New Features:
- **Camera extraction utility**: Robust camera number parsing from file names using regex
- **Camera split mapping**: Selects cameras systematically (first 5 for training, next 3 for testing)
- **Split-aware loading**: Only loads specific cameras based on split type

#### Key Functions:
```python
extract_camera_number(extr_name):  # Robust camera number extraction
get_camera_split_mapping cam_extrinsics, split)  # Camera selection logic
load_images_path()  # Split-aware image loading
```

#### Camera Split Logic:
```python
# Camera 1-5: Training (first 5 cameras by number)
# Camera 6-8: Testing (next 3 cameras by number)

if split == "train":
    return all_camera_idxs[:5]  # First 5 cameras
else:  # test
    return all_camera_idxs[5:8]  # Next 3 cameras
```

### 2. Dataset Readers (`2d-gaussian-splatting/scene/dataset_readers.py`)

#### Fixed Issues:
- **Bug fix**: Test dataset was incorrectly using train dataset poses (line 319)
- **Better error handling**: Image loading failures with proper error messages
- **Improved logging**: Clear indication of data loading process

### 3. Training Script (`2d-gaussian-splatting/train_with_charts_.py`)

#### Key Improvements:
- **Camera split summary**: Shows training/testing camera details at startup
- **Training cameras only**: Charts priors built only from training cameras
- **Enhanced validation**: Checks for valid training cameras before proceeding

#### Logging Output:
```
[CAMERA SPLIT SUMMARY]
  Training cameras: 5
  Testing cameras: 3
  Training camera names:
    1: cam_0001_0001.jpg (path/to/train_camera_1.jpg)
    2: cam_0002_0001.jpg (path/to/train_camera_2.jpg)
    ...
  Testing camera names:
    1: cam_0006_0001.jpg (path/to/test_camera_1.jpg)
    2: cam_0007_0001.jpg (path/to/test_camera_2.jpg)
    ...
```

### 3. Training Script (`train_with_charts_.py`) - CRITICAL FIX

#### **CHARTS DATA FILTERING** - Fixed main issue: Using all cameras instead of training only
**Original Problem:** `get_gaussian_parameters_from_charts_data` used ALL cameras from `charts_data.npz`
**New Implementation:** Filters `charts_data` to only include training cameras

#### Implementation Details:
```python
# Extract training camera numbers from training cameras
for cam in train_cameras:
    train_camera_nums.add(extract_camera_number_from_path(cam.image_path))

# Filter charts_data tensors to only include training cameras
filtered_indices = [i for i in range(total_cameras) if (i+1) in train_camera_nums]

# Filter all camera-dimensional tensors
for key, tensor in charts_data.items():
    if key != 'scale_factor' and tensor.shape[0] == n_original_cameras:
        filtered_charts_data[key] = tensor[filtered_indices]
        print(f"[INFO] Filtered {key}: {tensor.shape} → {filtered_charts_data[key].shape}")
```

**Before fix**: `charts_data['pts'].shape = torch.Size([8, height, width, 3])` (all 8 cameras)
**After fix**: `filtered_charts_data['pts'].shape = torch.Size([5, height, width, 3])` (5 training cameras only)

#### Key Improvements:
- **Training-Camera Filtering**: Ensures Gaussian parameters are built exclusively from training data
- **Debug Logging**: Shows tensor shape transformations
- **Camera Index Matching**: Maps filtered charts data to specific training camera numbers
- **Comprehensive Validation**: Validates that charts data contains only the expected training cameras

### 4. Charts Building (`matcha/dm_scene/charts.py`)

#### Optimization:
- Charts built only from filtered training cameras used during prior building
- Always uses the same filtered training camera set for consistency
- Enhanced error handling and cameras filtering
- Improved logging for debugging camera issues

## Usage

### Configuration
The implementation automatically handles camera splitting. If you have fewer than 8 cameras, it will:
- Use first 5 cameras for training (if available)
- Use remaining cameras for testing
- Display warnings if insufficient cameras

### File Organization
Your dataset should follow this structure:
```
dataset_root/
├── sparse/
│   ├── images.bin           # COLMAP camera extrinsics
│   └── cameras.bin          # COLMAP camera intrinsics
├── cam01/                   # Camera 1 folder
│   ├── cam_0001_0001.jpg    # Frame 1
│   ├── cam_0001_0002.jpg    # Frame 2
│   └── ...
├── cam02/                   # Camera 2 folder
│   ├── cam_0002_0001.jpg
│   └── ...
├── cam03/                   # Camera 3 folder
│   └── ...
├── cam04/                   # Camera 4 folder
│   └── ...
├── cam05/                   # Camera 5 folder  (Training)
│   └── ...
├── cam06/                   # Camera 6 folder  (Testing)
│   └── ...
├── cam07/                   # Camera 7 folder  (Testing)
│   └── ...
└── cam08/                   # Camera 8 folder  (Testing)
    └── ...
```

### Training Command
```bash
python train.py -s /path/to/your/dataset -o /path/to/output --sfm_config unposed
```

## Validation

Run the test script to validate implementation:
```bash
python test_camera_split.py
```

Expected output:
```
✅ Camera splitting implementation validated successfully!
✅ Training will use 5 cameras
✅ Testing will use 3 cameras
✅ No camera overlap between train/test sets
```

## Technical Details

### Camera Selection Logic
1. **Sorting**: Cameras are sorted by their extracted numbers
2. **Splitting**: Fixed split at 5 cameras for training
3. **Handle overflow**: If <8 cameras, use available ones

### Charts Data Tensor Filtering
- **Critical Fix**: Filters chronological data to only include training cameras
- **Tensor Dimension Filtering**: Reduces camera dimension from [8, height, width, channels] to [5, height, width, channels]
- **Camera-to-Index Mapping**: Maps camera numbers to tensor indices
- **Validation Logging**: Shows tensor shape transformations for debugging

### Priors Building
- Charts are built only from filtered training cameras (prevents test contamination)
- Confidence and depth maps only use filtered training camera views
- Prior tensors use same camera filtering as Gaussian parameters

### Error Handling
- Camera folder detection with graceful degradation
- Corrupted image file handling
- Insufficient camera warnings
- Clear diagnostic messages

## Benefits

1. **Proper Evaluation**: Test cameras are truly unseen during training
2. **Data Contamination Prevention**: Critical fix prevents test data in Gaussian parameters
3. **Tensor-Level Filtering**: Chronicles data filtered at tensor dimension level
4. **Better Generalization**: Models trained on constrained views validate better
5. **Clear Separation**: No data mixing between train/test
6. **Flexible Configuration**: Adapts to different camera counts
7. **Robust Processing**: Handles missing folders and files
8. **Comprehensive Logging**: Shows camera assignments and tensor transformations

## Implementation Status

✅ MultipleView dataset camera splitting - COMPLETE
✅ Dataset readers camera separation - COMPLETE
✅ **CRTICAL FIX**: Charts data filtering (train cameras only) - COMPLETE
✅ Test camera isolation - COMPLETE
✅ **ENHANCED**: Charts priors built from filtered training cameras - COMPLETE
✅ Training script tensor filtering - COMPLETE
✅ Validation and logging - COMPLETE
✅ Error handling and warnings - COMPLETE
✅ Test script for validation - COMPLETE
✅ Documentation with tensor filtering details - COMPLETE

## Conclusion

The camera splitting implementation provides a robust, production-ready solution for multi-view datasets that ensures proper train-test separation for 8-camera scenarios (5 training + 3 testing). The system gracefully handles edge cases and provides comprehensive logging for debugging and validation.