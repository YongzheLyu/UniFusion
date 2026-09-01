import argparse
import pickle
import numpy as np
import rerun as rr
from pathlib import Path


def load_metadata(metadata_path: Path):
    """Load metadata from processed data directory.

    Args:
        metadata_path: Path to metadata.pkl file

    Returns:
        dict containing metadata
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    return metadata


def load_points_data(data_dir: Path, frame_indices: list, view_ids: list):
    """Load all point cloud data.

    Args:
        data_dir: Directory containing the processed point cloud files
        frame_indices: List of frame indices
        view_ids: List of view IDs

    Returns:
        dict: {frame_idx: {view_id: points_array}}
    """
    points_data = {}

    for frame_idx in frame_indices:
        frame_data = {}
        for view_id in view_ids:
            points_filename = f"frame_{frame_idx:05d}_{view_id}_points.npy"
            points_path = data_dir / points_filename

            if points_path.exists():
                points = np.load(points_path)
                frame_data[view_id] = points
                print(f"[INFO] Loaded {view_id} for frame {frame_idx}: {points.shape[0]} points")
            else:
                print(f"[WARN] Points file not found: {points_filename}")

        if frame_data:
            points_data[frame_idx] = frame_data

    return points_data


def visualize_points_data(data_dir: Path):
    """Visualize processed depth points data using rerun.

    Args:
        data_dir: Directory containing processed point cloud data
    """
    # Load metadata
    metadata_path = data_dir / 'metadata.pkl'
    metadata = load_metadata(metadata_path)

    n_timestamps = metadata['n_timestamps']
    view_ids = metadata['view_ids']
    frame_indices = metadata['frame_indices']
    scale_factor = metadata['scale_factor']
    stride = metadata['stride']

    print(f"[INFO] Visualizing processed data:")
    print(f"  - Number of timestamps: {n_timestamps}")
    print(f"  - Views: {view_ids}")
    print(f"  - Scale factor: {scale_factor}")
    print(f"  - Stride: {stride}")

    # Load all points data
    print(f"\n[INFO] Loading point cloud data...")
    points_data = load_points_data(data_dir, frame_indices, view_ids)

    if not points_data:
        print("[ERROR] No point cloud data found!")
        return

    # Initialize rerun
    rr.init("processed_depth_points", spawn=False)
    server_uri = rr.serve_grpc()
    print(f"[INFO] gRPC server started at {server_uri}")
    rr.serve_web_viewer(connect_to=server_uri)
    # Try different connection methods based on rerun version
    # try:
    #     # Modern rerun API
    #     rr.connect()
    #     print("[INFO] Connected to rerun viewer")
    # except AttributeError:
    #     try:
    #         # Older rerun API
    #         server_uri = rr.serve_grpc()
    #         print(f"[INFO] gRPC server started at {server_uri}")
    #         rr.serve_web_viewer(connect_to=server_uri)
    #     except AttributeError:
    #         # Fallback: just initialize without connecting
    #         print("[INFO] Rerun initialized. Please connect manually or check rerun version.")

    # Visualize data
    for t_idx, frame_idx in enumerate(frame_indices):
        if frame_idx not in points_data:
            continue

        print(f"[INFO] Visualizing timestamp {t_idx} (frame {frame_idx})")

        # Set time sequence
        rr.set_time_sequence("frame", t_idx)

        frame_data = points_data[frame_idx]

        # Visualize each view
        for view_id in view_ids:
            if view_id not in frame_data:
                continue

            points = frame_data[view_id]

            # Create a simple camera pose (identity for now)
            # In a real scenario, you'd need the actual camera poses
            cam_R = np.eye(3, dtype=np.float32)
            cam_t = np.zeros(3, dtype=np.float32)

            # Log points and camera
            rr.log(f"points/{view_id}", rr.Points3D(points, radii=0.02))
            rr.log(
                f"cameras/{view_id}",
                rr.Transform3D(translation=cam_t, mat3x3=cam_R),
            )

            print(f"  [INFO] Logged {view_id} with {points.shape[0]} points")

    print(f"\n[INFO] Finished visualizing {len(points_data)} timestamps. View in rerun viewer.")

    # Keep the server running
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description="Visualize processed depth points data with rerun.")
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing processed point cloud data (with metadata.pkl).",
    )

    args = parser.parse_args()

    # Validate input
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    metadata_path = args.data_dir / 'metadata.pkl'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Run visualization
    visualize_points_data(args.data_dir)


if __name__ == "__main__":
    main()
