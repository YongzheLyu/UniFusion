#!/usr/bin/env python3
"""
Generate poses_bounds.npy from cameras.json and PLY point cloud.

This script converts camera parameters from cameras.json and computes scene bounds
from a PLY point cloud to create a poses_bounds.npy file in LLFF format.

Usage:
    python generate_poses_bounds.py --data_dir /path/to/data

The script expects:
    - cameras.json in the data directory
    - points.ply in the data directory (optional, for auto bounds calculation)
"""

import numpy as np
import json
import argparse
import os
from pathlib import Path


def load_ply_points(ply_path):
    """
    Load 3D points from PLY file.

    Args:
        ply_path: Path to the PLY file

    Returns:
        points: Nx3 array of 3D points
    """
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return points


def compute_bounds_from_points(points, cam_poses, percentile=5):
    """
    Compute near and far bounds for each camera based on point cloud depth.

    Args:
        points: Nx3 array of 3D points
        cam_poses: List of 4x4 camera-to-world matrices
        percentile: Percentile for min/max depth filtering

    Returns:
        bounds: Nx2 array of [near, far] for each camera
    """
    bounds = []

    for pose in cam_poses:
        # Transform points to camera space
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Camera center in world coordinates
        cam_center = t

        # Compute depth (distance along camera z-axis)
        # Camera looks at -z direction in camera space
        cam_forward = -R[:, 2]  # Third column, negated

        # Distance from camera center to each point projected onto camera forward direction
        points_from_cam = points - cam_center[None, :]
        depths = np.dot(points_from_cam, cam_forward)

        # Filter out points behind camera
        valid_depths = depths[depths > 0]

        if len(valid_depths) > 0:
            near = np.percentile(valid_depths, percentile)
            far = np.percentile(valid_depths, 100 - percentile)
            # Add some margin
            near = max(near * 0.9, 0.01)
            far = far * 1.1
        else:
            # Default values if no valid points
            near = 0.01
            far = 10.0

        bounds.append([near, far])

    return np.array(bounds)


def cameras_json_to_poses_bounds(cameras_json_path, ply_path=None, output_path=None,
                                  default_near=0.01, default_far=10.0, hwf=None):
    """
    Convert cameras.json to poses_bounds.npy format.

    Args:
        cameras_json_path: Path to cameras.json
        ply_path: Path to points.ply (optional, for automatic bounds)
        output_path: Path to save poses_bounds.npy (default: same dir as cameras.json)
        default_near: Default near plane if no PLY provided
        default_far: Default far plane if no PLY provided
        hwf: (height, width, focal) tuple. If None, will use dummy values

    Returns:
        poses_bounds: Nx17 array in LLFF format
    """
    # Load cameras.json
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)

    cams2world = cameras_data['cams2world']
    focals = cameras_data['focals']
    filepaths = cameras_data.get('filepaths', [])

    n_cams = len(cams2world)

    # Convert to numpy arrays
    poses = []
    for c2w in cams2world:
        pose = np.array(c2w)  # 4x4 matrix
        poses.append(pose)

    # Compute bounds
    if ply_path is not None and os.path.exists(ply_path):
        print(f"Loading point cloud from {ply_path}")
        try:
            points = load_ply_points(ply_path)
            print(f"Loaded {len(points)} points")
            bounds = compute_bounds_from_points(points, poses)
            print(f"Computed bounds from point cloud")
        except Exception as e:
            print(f"Warning: Could not load PLY file: {e}")
            print(f"Using default bounds: near={default_near}, far={default_far}")
            bounds = np.array([[default_near, default_far]] * n_cams)
    else:
        print(f"No PLY file provided, using default bounds: near={default_near}, far={default_far}")
        bounds = np.array([[default_near, default_far]] * n_cams)

    # Get image dimensions and focal length
    # If hwf not provided, try to infer from images or use defaults
    if hwf is None:
        if filepaths and len(filepaths) > 0:
            # Try to load first image to get dimensions
            try:
                from PIL import Image
                # Get absolute path
                img_path = filepaths[0]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(os.path.dirname(cameras_json_path), img_path)

                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    h, w = img.height, img.width
                    f = focals[0]
                    print(f"Inferred image size from {img_path}: {h}x{w}, focal={f}")
                else:
                    # Use default values
                    h, w, f = 480, 640, focals[0]
                    print(f"Warning: Could not find image, using default size: {h}x{w}, focal={f}")
            except Exception as e:
                print(f"Warning: Could not load image: {e}")
                h, w, f = 480, 640, focals[0]
                print(f"Using default size: {h}x{w}, focal={f}")
        else:
            h, w, f = 480, 640, focals[0] if focals else 500.0
            print(f"Using default size: {h}x{w}, focal={f}")
    else:
        h, w, f = hwf
        print(f"Using provided size: {h}x{w}, focal={f}")

    # Build poses_bounds array
    # LLFF format: each row is [R (3x3 flattened), t (3x1), h, w, f, near, far]
    # Total: 3*5 + 2 = 17 elements per camera
    poses_bounds = []

    for i in range(n_cams):
        pose = poses[i]
        R = pose[:3, :3]  # 3x3 rotation
        t = pose[:3, 3]   # 3x1 translation

        # LLFF stores poses as [down, right, backwards] = [R|t]
        # where each column is a direction in world space
        # Reshape to 3x5: [R | t | [h, w, f]^T]
        pose_hwf = np.concatenate([R, t[:, None],
                                   np.array([[h], [w], [f]])], axis=1)  # 3x5

        # Flatten to 15 elements
        pose_flat = pose_hwf.reshape(-1)

        # Add near and far bounds
        near, far = bounds[i]
        pose_bounds = np.concatenate([pose_flat, [near, far]])

        poses_bounds.append(pose_bounds)

    poses_bounds = np.array(poses_bounds)  # Nx17

    # Save output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(cameras_json_path), 'poses_bounds.npy')

    np.save(output_path, poses_bounds)
    print(f"Saved poses_bounds.npy to {output_path}")
    print(f"Shape: {poses_bounds.shape}")
    print(f"Bounds range: near=[{bounds[:, 0].min():.3f}, {bounds[:, 0].max():.3f}], "
          f"far=[{bounds[:, 1].min():.3f}, {bounds[:, 1].max():.3f}]")

    return poses_bounds


def main():
    parser = argparse.ArgumentParser(description='Generate poses_bounds.npy from cameras.json')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing cameras.json')
    parser.add_argument('--cameras_json', type=str, default='cameras.json',
                        help='Name of cameras JSON file (default: cameras.json)')
    parser.add_argument('--ply_file', type=str, default='points.ply',
                        help='Name of PLY file for bounds calculation (default: points.ply)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for poses_bounds.npy (default: data_dir/poses_bounds.npy)')
    parser.add_argument('--near', type=float, default=0.01,
                        help='Default near plane (default: 0.01)')
    parser.add_argument('--far', type=float, default=10.0,
                        help='Default far plane (default: 10.0)')
    parser.add_argument('--height', type=int, default=288,
                        help='Image height (default: infer from images)')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width (default: infer from images)')

    args = parser.parse_args()

    # Build paths
    cameras_json_path = os.path.join(args.data_dir, args.cameras_json)
    ply_path = os.path.join(args.data_dir, args.ply_file)

    if not os.path.exists(cameras_json_path):
        print(f"Error: {cameras_json_path} not found")
        return

    if not os.path.exists(ply_path):
        print(f"Warning: {ply_path} not found, will use default bounds")
        ply_path = None

    # Set hwf if provided
    hwf = None
    if args.height is not None and args.width is not None:
        # Focal will be taken from cameras.json
        with open(cameras_json_path, 'r') as f:
            cameras_data = json.load(f)
        focal = cameras_data['focals'][0]
        hwf = (args.height, args.width, focal)

    # Generate poses_bounds.npy
    poses_bounds = cameras_json_to_poses_bounds(
        cameras_json_path,
        ply_path=ply_path,
        output_path=args.output,
        default_near=args.near,
        default_far=args.far,
        hwf=hwf
    )


if __name__ == '__main__':
    main()
