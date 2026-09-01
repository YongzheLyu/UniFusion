#!/usr/bin/env python3
"""
将 pose_metadata.json 格式转换为 COLMAP 二进制格式

输入 JSON 格式:
{
  "camera_names": ["cooking_undist_cam00", ...],
  "poses": [[4x4 cam2world], ...],
  "intrinsics": [[3x3], ...],
  "focals": [fx, ...]
}

输出 COLMAP 格式:
- cameras.bin: 相机内参
- images.bin: 相机外参 (world2cam)
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from colmap.read_write_model import Camera, Image, write_cameras_binary, write_images_binary, rotmat2qvec


def rotmat2qvec(R):
    """
    将旋转矩阵转换为四元数

    使用 SciPy 的 Rotation 类，格式为 [qx, qy, qz, qw]
    """
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    # COLMAP 格式是 [qw, qx, qy, qz]，但 Rotation 返回 [x, y, z, w]
    # 实际上 COLMAP 的 qvec 是 [qw, qx, qy, qz]
    return np.array([q[3], q[0], q[1], q[2]])


def convert_pose_metadata_to_colmap(
    json_path: Path,
    output_dir: Path,
    model_id: int = 0,
    image_file_pattern: str = None
):
    """
    将 pose_metadata.json 转换为 COLMAP 格式

    Args:
        json_path: 输入的 JSON 文件路径
        output_dir: 输出目录，将创建 sparse/0/ 子目录
        model_id: COLMAP model ID
        image_file_pattern: 图像文件名模式，如 "cam{cam_idx:02d}_{frame_idx:04d}.jpg"
    """
    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_names = data['camera_names']
    poses = np.array(data['poses'])  # [N, 4, 4] cam2world
    intrinsics = np.array(data['intrinsics'])  # [N, 3, 3]
    focals = data['focals']

    n_cameras = len(camera_names)
    print(f"[INFO] 找到 {n_cameras} 个相机")

    # 创建输出目录
    sparse_dir = output_dir / "sparse" / str(model_id)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # 准备 COLMAP 数据
    cameras = {}  # {camera_id: Camera}
    images = {}   # {image_id: Image}

    for cam_idx, camera_name in enumerate(camera_names):
        camera_id = cam_idx + 1  # camera_id 从 1 开始
        image_id = cam_idx + 1    # image_id 从 1 开始

        # 提取内参
        K = intrinsics[cam_idx]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # 创建 COLMAP Camera 对象 (PINHOLE 模型)
        cameras[camera_id] = Camera(
            id=camera_id,
            model='PINHOLE',
            width=int(data.get('image_width', 512)),
            height=int(data.get('image_height', 288)),
            params=np.array([fx, fy, cx, cy])
        )

        # 计算 world2cam (COLMAP 使用的格式)
        # pose 是 cam2world，所以 world2cam = inverse(cam2world)
        cam2world = poses[cam_idx]
        world2cam = np.linalg.inv(cam2world)

        # 提取旋转和平移
        R_w2c = world2cam[:3, :3]
        t_w2c = world2cam[:3, 3]

        # 转换为四元数
        qvec = rotmat2qvec(R_w2c)

        # 确定图像文件名
        if image_file_pattern is not None:
            # 使用模式生成文件名
            # 尝试从 camera_name 中提取相机编号
            # 如 "cooking_undist_cam00" -> 0
            if 'cam' in camera_name:
                cam_num_str = camera_name.rsplit('cam', 1)[-1]
                try:
                    cam_num = int(cam_num_str)
                except ValueError:
                    cam_num = cam_idx
            else:
                cam_num = cam_idx

            # 帧号设为 0（因为是静态 pose）
            frame_idx = 0
            image_name = image_file_pattern.format(cam_idx=cam_num, frame_idx=frame_idx)
        else:
            # 直接使用 camera_name.jpg
            image_name = f"{camera_name}.jpg"

        # 创建 COLMAP Image 对象
        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=t_w2c,
            camera_id=camera_id,
            name=image_name,
            xys=np.zeros((0, 2)),
            point3D_ids=np.zeros((0,), dtype=np.int64)
        )

    # 写入 COLMAP 二进制文件
    cameras_bin_path = sparse_dir / "cameras.bin"
    images_bin_path = sparse_dir / "images.bin"

    write_cameras_binary(cameras, str(cameras_bin_path))
    write_images_binary(images, str(images_bin_path))

    print(f"[INFO] 已写入: {cameras_bin_path}")
    print(f"[INFO] 已写入: {images_bin_path}")

    # 创建空的 points3D.bin
    points3D_bin_path = sparse_dir / "points3D.bin"
    points3D_bin_path.write_bytes(b'')

    print(f"[INFO] 已写入: {points3D_bin_path}")

    return cameras, images


def main():
    parser = argparse.ArgumentParser(
        description="将 pose_metadata.json 转换为 COLMAP 格式"
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="输入的 pose_metadata.json 文件路径"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="输出目录 (默认: 在 json 同级目录下创建 sparse/)"
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=0,
        help="COLMAP model ID (默认: 0)"
    )
    parser.add_argument(
        "--image-pattern",
        type=str,
        default=None,
        help="图像文件名模式，如 'cam{cam_idx:02d}_{frame_idx:04d}.jpg' "
             "(变量: cam_idx, frame_idx, 默认使用 camera_name.jpg)"
    )

    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir is None:
        args.output_dir = args.json_path.parent

    print(f"\n{'='*60}")
    print("Pose Metadata 到 COLMAP 转换")
    print(f"{'='*60}")
    print(f"输入: {args.json_path}")
    print(f"输出: {args.output_dir}")
    print(f"{'='*60}\n")

    convert_pose_metadata_to_colmap(
        json_path=args.json_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        image_file_pattern=args.image_pattern
    )

    print(f"\n{'='*60}")
    print("转换完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
