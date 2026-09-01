#!/usr/bin/env python3
"""
Convert EasyMocap camera parameters to COLMAP format for use with MAtCha.

Usage:
    python convert_easymocap_to_colmap.py \
        --input /path/to/easymocap_multi/all_people \
        --output /path/to/colmap_dataset
"""

import os
import sys
import json
import struct
import shutil
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import collections

# COLMAP data structures
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

# COLMAP camera models
CAMERA_MODEL_NAMES = {
    "PINHOLE": CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
}


def rotmat2qvec(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """Write bytes to a binary file."""
    if isinstance(data, (list, tuple)):
        bytes_to_write = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes_to_write = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes_to_write)


def write_cameras_binary(cameras, path_to_model_file):
    """Write cameras to COLMAP binary format."""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for camera in cameras.values():
            write_next_bytes(fid, camera.id, "I")
            write_next_bytes(fid, camera.model.model_id, "I")
            write_next_bytes(fid, camera.width, "Q")
            write_next_bytes(fid, camera.height, "Q")
            for param in camera.params:
                write_next_bytes(fid, param, "d")


def write_images_binary(images, path_to_model_file):
    """Write images to COLMAP binary format."""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for img in images.values():
            write_next_bytes(fid, img.id, "I")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "I")
            # Write image name
            name_bytes = img.name.encode("utf-8")
            for char in name_bytes:
                write_next_bytes(fid, bytes([char]), "c")
            write_next_bytes(fid, b"\x00", "c")
            # Write 2D points (empty for now)
            write_next_bytes(fid, len(img.xys), "Q")
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, xy.tolist(), "dd")
                write_next_bytes(fid, point3D_id, "q")


def write_points3D_binary(points3D, path_to_model_file):
    """Write 3D points to COLMAP binary format (empty for now)."""
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for point3D in points3D.values():
            write_next_bytes(fid, point3D.id, "Q")
            write_next_bytes(fid, point3D.xyz.tolist(), "ddd")
            write_next_bytes(fid, point3D.rgb.tolist(), "BBB")
            write_next_bytes(fid, point3D.error, "d")
            write_next_bytes(fid, len(point3D.image_ids), "Q")
            for image_id, point2D_idx in zip(point3D.image_ids, point3D.point2D_idxs):
                write_next_bytes(fid, image_id, "I")
                write_next_bytes(fid, point2D_idx, "I")


def convert_easymocap_to_colmap(input_path, output_path):
    """Convert EasyMocap camera parameters to COLMAP format (per-timestep datasets)."""

    # Load EasyMocap cameras
    cameras_json_path = os.path.join(input_path, "cameras.json")
    with open(cameras_json_path, 'r') as f:
        easymocap_data = json.load(f)

    # Get camera names
    camera_names = easymocap_data["filepaths"]
    print(f"Found {len(camera_names)} cameras: {camera_names}")

    # Determine number of frames by checking first camera
    first_cam_images_dir = Path(input_path) / "images" / camera_names[0]
    if not first_cam_images_dir.exists():
        print(f"Error: Image directory {first_cam_images_dir} does not exist!")
        sys.exit(1)

    image_files = sorted(list(first_cam_images_dir.glob("*.jpg")) +
                        list(first_cam_images_dir.glob("*.png")))
    num_frames = len(image_files)
    print(f"Found {num_frames} frames")

    # Validate all cameras have same number of frames
    for cam_name in camera_names:
        cam_images_dir = Path(input_path) / "images" / cam_name
        if not cam_images_dir.exists():
            print(f"Error: Image directory {cam_images_dir} does not exist!")
            sys.exit(1)
        cam_image_files = sorted(list(cam_images_dir.glob("*.jpg")) +
                                list(cam_images_dir.glob("*.png")))
        if len(cam_image_files) != num_frames:
            print(f"Error: {cam_name} has {len(cam_image_files)} frames, expected {num_frames}")
            sys.exit(1)

    # Create output base directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare camera data (same for all frames)
    colmap_cameras = {}
    camera_poses = {}  # Store poses for each camera

    for cam_idx, cam_name in enumerate(camera_names):
        cam_data = easymocap_data[cam_idx]

        # Extract intrinsics
        K = np.array(cam_data["intrinsic"])
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        width = cam_data["width"]
        height = cam_data["height"]

        # Create COLMAP camera (one per physical camera)
        camera_id = cam_idx + 1
        colmap_cameras[camera_id] = Camera(
            id=camera_id,
            model=CAMERA_MODEL_NAMES["PINHOLE"],
            width=width,
            height=height,
            params=np.array([fx, fy, cx, cy])
        )

        # Extract extrinsics (world-to-camera transformation)
        w2c_matrix = np.array(cam_data["extrinsic"])
        R_w2c = w2c_matrix[:3, :3]
        t_w2c = w2c_matrix[:3, 3]

        # Convert rotation matrix to quaternion
        qvec = rotmat2qvec(R_w2c)
        tvec = t_w2c

        camera_poses[cam_name] = {
            'camera_id': camera_id,
            'qvec': qvec,
            'tvec': tvec
        }

    # Process each frame (timestep)
    print("\nProcessing frames...")
    for frame_idx in tqdm(range(num_frames), desc="Converting frames"):
        # Create frame-specific output directories
        frame_name = f"frame_{frame_idx:04d}"
        frame_output = output_path / frame_name
        frame_images_dir = frame_output / "images"
        frame_sparse_dir = frame_output / "sparse" / "0"
        frame_images_dir.mkdir(parents=True, exist_ok=True)
        frame_sparse_dir.mkdir(parents=True, exist_ok=True)

        # Prepare COLMAP data structures for this frame
        colmap_images = {}

        # Process each camera for this frame
        for cam_idx, cam_name in enumerate(camera_names):
            cam_images_dir = Path(input_path) / "images" / cam_name

            # Get the specific image for this frame
            frame_images = sorted(list(cam_images_dir.glob("*.jpg")) +
                                 list(cam_images_dir.glob("*.png")))
            src_image = frame_images[frame_idx]

            # New image name with camera prefix
            new_image_name = f"{cam_name}_{src_image.name}"

            # Copy image to frame output directory
            dst_path = frame_images_dir / new_image_name
            shutil.copy2(src_image, dst_path)

            # Create COLMAP image entry
            image_id = cam_idx + 1  # 1-4 for each frame
            pose = camera_poses[cam_name]

            colmap_images[image_id] = Image(
                id=image_id,
                qvec=pose['qvec'],
                tvec=pose['tvec'],
                camera_id=pose['camera_id'],
                name=new_image_name,
                xys=np.array([]),  # No 2D points
                point3D_ids=np.array([], dtype=np.int32)  # No 3D point associations
            )

        # Write COLMAP binary files for this frame
        colmap_points3D = {}  # Empty points for all frames
        write_cameras_binary(colmap_cameras, frame_sparse_dir / "cameras.bin")
        write_images_binary(colmap_images, frame_sparse_dir / "images.bin")
        write_points3D_binary(colmap_points3D, frame_sparse_dir / "points3D.bin")

    print(f"\nConversion complete!")
    print(f"Created {num_frames} COLMAP datasets in: {output_path}")
    print(f"Each frame directory (frame_0000/, frame_0001/, ...) contains:")
    print(f"  - 4 images (one per camera)")
    print(f"  - COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)")
    print(f"\nTo use with MAtCha, run:")
    print(f"  python train.py -s {output_path}/frame_0000 --sfm_config posed --n_images 4")
    print(f"\nTo batch process all frames, you can loop over frame directories:")


def main():
    parser = argparse.ArgumentParser(description="Convert EasyMocap to COLMAP format")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to EasyMocap data (containing cameras.json and images/)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output COLMAP dataset")
    args = parser.parse_args()

    # Check input path exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist!")
        sys.exit(1)

    if not (input_path / "cameras.json").exists():
        print(f"Error: cameras.json not found in {input_path}!")
        sys.exit(1)

    # Convert
    convert_easymocap_to_colmap(args.input, args.output)


if __name__ == "__main__":
    main()
