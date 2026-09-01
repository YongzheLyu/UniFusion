import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import pickle
import numpy as np
from pathlib import Path


def depth_to_points_simple(depth: np.ndarray, focal: float = None, stride: int = 1):
    """Simple back-projection of depth map to 3D points using estimated camera parameters.

    Args:
        depth: Depth map (H, W)
        focal: Focal length (if None, estimated from image size)
        stride: Subsampling stride

    Returns:
        points: (N, 3) array of 3D points
    """
    H, W = depth.shape

    if focal is None:
        focal = min(H, W) * 0.7  # Conservative focal length estimate

    cx, cy = W * 0.5, H * 0.5

    # Apply stride
    if stride > 1:
        depth = depth[::stride, ::stride]
        H, W = depth.shape

    # Pixel coordinates
    xs = np.arange(W, dtype=np.float32) * stride
    ys = np.arange(H, dtype=np.float32) * stride
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Camera rays (normalized)
    x_cam = (grid_x - cx) / focal
    y_cam = (grid_y - cy) / focal
    rays_cam = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=-1)  # (H, W, 3)

    # Scale by depth to get camera-frame points
    pts_cam = rays_cam * depth[..., None]  # (H, W, 3)
    pts_world = pts_cam.reshape(-1, 3)  # (N, 3)

    return pts_world


def process_depth_points(preprocessed_path: Path, output_dir: Path, stride: int = 1,
                        view_ids: list = None, max_frames: int = None):
    """Process preprocessed temporal data and save depth points to files.

    Args:
        preprocessed_path: Path to preprocessed data pickle file
        output_dir: Directory to save processed point cloud data
        stride: Subsampling factor for depth points
        view_ids: List of view IDs to process (None = all views)
        max_frames: Maximum number of frames to process (None = all)
    """
    # Load preprocessed data
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed data file not found: {preprocessed_path}")

    print(f"[INFO] Loading preprocessed data from {preprocessed_path}")
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)

    print(f"[INFO] Loaded data keys: {list(data.keys())}")

    # Extract data
    temporal_reference_data = data['temporal_reference_data']  # List of depth tensors per timestamp
    temporal_frame_indices = data['temporal_frame_indices']    # Frame indices
    scale_factor = data.get('scale_factor', 1.0)               # Scale factor

    n_timestamps = len(temporal_reference_data)
    n_views_per_timestamp = temporal_reference_data[0].shape[0] if n_timestamps > 0 else 0

    print(f"[INFO] Data overview:")
    print(f"  - Number of timestamps: {n_timestamps}")
    print(f"  - Views per timestamp: {n_views_per_timestamp}")
    print(f"  - Scale factor: {scale_factor}")
    print(f"  - Frame indices: {temporal_frame_indices[:min(5, len(temporal_frame_indices))]}..."
          if len(temporal_frame_indices) > 5 else f"  - Frame indices: {temporal_frame_indices}")

    # Limit frames if requested
    if max_frames is not None and max_frames < n_timestamps:
        temporal_reference_data = temporal_reference_data[:max_frames]
        temporal_frame_indices = temporal_frame_indices[:max_frames]
        n_timestamps = max_frames
        print(f"[INFO] Limited to first {max_frames} frames")

    # Generate view IDs if not provided
    if view_ids is None:
        view_ids = [f"cam{i:02d}" for i in range(n_views_per_timestamp)]
    else:
        # Filter to available views
        available_view_ids = [f"cam{i:02d}" for i in range(n_views_per_timestamp)]
        view_ids = [vid for vid in view_ids if vid in available_view_ids]
        if len(view_ids) == 0:
            print(f"[WARN] No matching view IDs found. Available: {available_view_ids}")
            return

    print(f"[INFO] Processing {len(view_ids)} views: {', '.join(view_ids)}")
    print(f"[INFO] Using stride {stride} for depth point subsampling")
    print(f"[INFO] Saving to directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        'n_timestamps': n_timestamps,
        'n_views': len(view_ids),
        'view_ids': view_ids,
        'frame_indices': temporal_frame_indices,
        'scale_factor': scale_factor,
        'stride': stride,
        'original_preprocessed_file': str(preprocessed_path)
    }

    metadata_path = output_dir / 'metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"[INFO] Saved metadata to {metadata_path}")

    # Process each timestamp
    for t_idx, (frame_idx, ref_data) in enumerate(zip(temporal_frame_indices, temporal_reference_data)):
        print(f"[INFO] Processing timestamp {t_idx} (frame {frame_idx})")

        # Process each view
        for view_idx, view_id in enumerate(view_ids):
            if view_idx >= ref_data.shape[0]:
                continue

            # Get depth map for this view: (H, W)
            depth_map = ref_data[view_idx].cpu().numpy()

            # Back-project depth to 3D points
            pts_world = depth_to_points_simple(depth_map, stride=stride)

            # Apply scale factor
            pts_world = pts_world * scale_factor

            # Save points to file
            points_filename = f"frame_{frame_idx:05d}_{view_id}_points.npy"
            points_path = output_dir / points_filename
            np.save(points_path, pts_world.astype(np.float32))

            print(f"  [INFO] Saved {view_id} with {pts_world.shape[0]} points to {points_filename}")

    print(f"\n[INFO] Finished processing {n_timestamps} timestamps.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Total files saved: {len(view_ids) * n_timestamps} point cloud files + 1 metadata file")


def main():
    parser = argparse.ArgumentParser(description="Process preprocessed temporal data and save depth points.")
    parser.add_argument(
        "preprocessed_data",
        type=Path,
        help="Path to preprocessed temporal data pickle file. Should contain temporal_reference_data, "
             "temporal_frame_indices, and scale_factor.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to save processed point cloud data.",
    )
    parser.add_argument(
        "--view",
        type=str,
        nargs="+",
        default=None,
        help="View ID(s) to process. If not specified, processes all available views. "
             "Format: cam00 cam01 cam02 etc.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Subsampling factor for depth points (default: 2)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.preprocessed_data.exists():
        raise FileNotFoundError(f"Preprocessed data file not found: {args.preprocessed_data}")

    # Run processing
    process_depth_points(
        args.preprocessed_data,
        args.output_dir,
        stride=args.stride,
        view_ids=args.view,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
