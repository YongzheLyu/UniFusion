#!/usr/bin/env python3
"""
将 gopro_calibs.csv 格式的相机参数转换为 pose_metadata.json 格式

CSV 格式:
cam_uid,graph_uid,tx_world_cam,ty_world_cam,tz_world_cam,qx_world_cam,qy_world_cam,qz_world_cam,qw_world_cam,
image_width,image_height,intrinsics_type,intrinsics_0(fx),intrinsics_1(fy),intrinsics_2(cx),
intrinsics_3(cy),intrinsics_4-7(畸变参数),...

输出 JSON 格式:
{
  "sequence": "...",
  "sequence_key": "...",
  "target": "...",
  "camera_count": N,
  "camera_names": ["seq_cam00", "seq_cam01", ...],
  "poses": [[4x4 cam2world], ...],
  "intrinsics": [[3x3], ...],
  "image_size": 512,
  "focals": [fx, ...]
}
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    将四元数转换为旋转矩阵

    四元数格式: [qx, qy, qz, qw]
    旋转矩阵: world_to_cam
    """
    # 归一化四元数
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    x, y, z, w = qx/norm, qy/norm, qz/norm, qw/norm

    # 转换为旋转矩阵 (world_to_cam)
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R


def convert_csv_to_pose_metadata(
    csv_path: Path,
    output_path: Path,
    sequence_name: str = None,
    target_name: str = None,
    image_size: int = 512
) -> Dict[str, Any]:
    """
    将 CSV 文件转换为 pose_metadata.json 格式

    Args:
        csv_path: 输入的 CSV 文件路径
        output_path: 输出的 JSON 文件路径
        sequence_name: 序列名称 (如果为 None，从 csv_path 推断)
        target_name: 目标名称 (如果为 None，从 sequence_name 推断)
        image_size: 图像大小 (用于内参缩放)

    Returns:
        转换后的数据字典
    """
    if sequence_name is None:
        # 从路径中推断序列名，如 "_cooking" -> "cooking"
        sequence_name = csv_path.parent.name
        if sequence_name.startswith('_'):
            sequence_name = sequence_name[1:]

    if target_name is None:
        target_name = sequence_name

    # 读取 CSV 文件
    cameras = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cameras.append(row)

    if not cameras:
        raise ValueError(f"CSV 文件为空: {csv_path}")

    n_cameras = len(cameras)

    # 从第一个相机获取图像尺寸
    img_width = int(cameras[0]['image_width'])
    img_height = int(cameras[0]['image_height'])

    print(f"[INFO] 找到 {n_cameras} 个相机")
    print(f"[INFO] 图像尺寸: {img_width}x{img_height}")
    print(f"[INFO] 目标 image_size: {image_size}")

    # 准备输出数据
    camera_names = []
    poses = []  # 4x4 cam2world 矩阵
    intrinsics = []  # 3x3 内参矩阵
    focals = []

    for cam_data in cameras:
        cam_uid = cam_data['cam_uid']

        # 生成相机名称，如 "cooking_undist_cam00"
        cam_num = int(cam_uid.replace('cam', ''))
        camera_name = f"{target_name}_undist_cam{cam_num:02d}"
        camera_names.append(camera_name)

        # 解析位置和四元数
        tx = float(cam_data['tx_world_cam'])
        ty = float(cam_data['ty_world_cam'])
        tz = float(cam_data['tz_world_cam'])
        qx = float(cam_data['qx_world_cam'])
        qy = float(cam_data['qy_world_cam'])
        qz = float(cam_data['qz_world_cam'])
        qw = float(cam_data['qw_world_cam'])

        # 解析内参
        fx = float(cam_data['intrinsics_0'])
        fy = float(cam_data['intrinsics_1'])
        cx = float(cam_data['intrinsics_2'])
        cy = float(cam_data['intrinsics_3'])

        # 计算缩放因子
        scale = image_size / max(img_width, img_height)

        # 缩放内参
        fx_scaled = fx * scale
        fy_scaled = fy * scale
        cx_scaled = cx * scale
        cy_scaled = cy * scale

        # 构建 cam2world 矩阵
        # world_to_cam 的旋转矩阵和平移向量
        R_w2c = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        t_w2c = np.array([tx, ty, tz])

        # cam2world = world_to_cam 的逆
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_w2c

        T_c2w = np.linalg.inv(T_w2c)
        poses.append(T_c2w.tolist())

        # 构建内参矩阵
        K = np.array([
            [fx_scaled, 0.0, cx_scaled],
            [0.0, fy_scaled, cy_scaled],
            [0.0, 0.0, 1.0]
        ])
        intrinsics.append(K.tolist())
        focals.append(fx_scaled)

    # 计算输出图像的高度
    aspect_ratio = img_height / img_width
    image_height_scaled = int(image_size * aspect_ratio)

    # 构建完整输出
    output_data = {
        "sequence": f"_{sequence_name}",
        "sequence_key": sequence_name,
        "target": target_name,
        "camera_count": n_cameras,
        "camera_names": camera_names,
        "poses": poses,
        "intrinsics": intrinsics,
        "image_size": image_size,
        "image_width": image_size,
        "image_height": image_height_scaled,
        "focals": focals,
        "model_name": "MASt3R",
        "source_file": str(csv_path)
    }

    # 写入 JSON 文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"[INFO] 已保存到: {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="将 gopro_calibs.csv 转换为 pose_metadata.json 格式"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="输入的 CSV 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
            help="输出的 JSON 文件路径 (默认: 在数据目录下生成 pose_metadata.json)"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="序列名称 (如果为 None，从文件路径推断)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="目标名称 (如果为 None，从序列名推断)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="目标图像大小 (默认: 512)"
    )

    args = parser.parse_args()

    # 确定输出路径
    if args.output is None:
        args.output = args.csv_path.parent / "pose_metadata.json"

    print(f"\n{'='*60}")
    print("CSV 到 Pose Metadata 转换")
    print(f"{'='*60}")
    print(f"输入: {args.csv_path}")
    print(f"输出: {args.output}")
    print(f"{'='*60}\n")

    convert_csv_to_pose_metadata(
        csv_path=args.csv_path,
        output_path=args.output,
        sequence_name=args.sequence,
        target_name=args.target,
        image_size=args.image_size
    )

    print(f"\n{'='*60}")
    print("转换完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
