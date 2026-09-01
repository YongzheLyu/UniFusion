#!/usr/bin/env python3
"""
完整的数据集处理流程脚本
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict
import re
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def natural_key(path: Path) -> List[object]:
    """用于自然排序的 key"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.stem)]


def convert_colmap_to_pose_metadata(colmap_dir: Path, output_path: Path, image_size: int = 512,
                                    camera_ids: Optional[List[int]] = None,
                                    min_resolution: int = 2000,
                                    use_camera_index_as_name: bool = True) -> Dict:
    """将 COLMAP 格式的 cameras.txt 和 images.txt 转换为 pose_metadata.json 格式

    Args:
        colmap_dir: 包含 cameras.txt 和 images.txt 的目录
        output_path: 输出 JSON 文件路径
        image_size: 目标图像宽度
        camera_ids: 指定要使用的相机ID列表（如果为 None，则使用所有满足条件的相机）
        min_resolution: 最小分辨率阈值（宽度或高度小于此值的相机被视为 ego cameras 并被排除）
        use_camera_index_as_name: 如果为 True，使用相机索引作为 camera_name，避免打乱顺序
    """
    cameras_file = colmap_dir / "cameras.txt"
    images_file = colmap_dir / "images.txt"

    if not cameras_file.exists():
        raise FileNotFoundError(f"找不到 cameras.txt: {cameras_file}")
    if not images_file.exists():
        raise FileNotFoundError(f"找不到 images.txt: {images_file}")

    # 读取相机内参
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            cameras[camera_id] = {
                'id': camera_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }

    # 根据 name 前缀识别 ego 和 exo cameras
    ego_camera_ids = []
    exo_camera_ids = []

    with open(images_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 10:
                # 第一行：位姿数据
                camera_id = int(parts[8])
                name = parts[9]

                # 根据前缀判断 ego/exo
                if name.startswith('aria'):
                    if camera_id not in ego_camera_ids:
                        ego_camera_ids.append(camera_id)
                elif name.startswith('cam'):
                    if camera_id not in exo_camera_ids:
                        exo_camera_ids.append(camera_id)

    ego_camera_ids = sorted(ego_camera_ids)
    exo_camera_ids = sorted(exo_camera_ids)

    print(f"[INFO] 根据名称前缀识别的 ego cameras: {ego_camera_ids}")
    print(f"[INFO] 根据名称前缀识别的 exo cameras: {exo_camera_ids}")

    # 读取图像位姿（每张图片两行：位姿行 + 2D点行）
    images = []
    with open(images_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()

            # 检查是否是位姿行（第一列是整数ID）
            if len(parts) >= 10:
                try:
                    image_id = int(parts[0])
                    # 确认是位姿行，不是2D点行（2D点行第一列可能是浮点数）
                    qw = float(parts[1])
                    qx = float(parts[2])
                    qy = float(parts[3])
                    qz = float(parts[4])
                    tx = float(parts[5])
                    ty = float(parts[6])
                    tz = float(parts[7])
                    camera_id = int(parts[8])
                    name = parts[9]

                    # 只记录 exo cameras 的图像
                    if name.startswith('cam'):
                        images.append({
                            'id': image_id,
                            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                            'tx': tx, 'ty': ty, 'tz': tz,
                            'camera_id': camera_id,
                            'name': name
                        })
                except (ValueError, IndexError) as e:
                    # 跳过格式错误的行（可能是2D点行或损坏数据）
                    pass

    # 使用根据名称前缀识别的 exo cameras
    print(f"[INFO] 使用 {len(exo_camera_ids)} 个 exo cameras (ID: {exo_camera_ids})")

    if camera_ids is not None:
        # camera_ids 是 exo cameras 的索引 (0-based)
        selected_colmap_ids = []
        for idx in camera_ids:
            if 0 <= idx < len(exo_camera_ids):
                selected_colmap_ids.append(exo_camera_ids[idx])
            else:
                print(f"[WARNING] 索引 {idx} 超出范围，exo cameras 共 {len(exo_camera_ids)} 个")
        print(f"[INFO] 根据索引 {camera_ids} 选择的 COLMAP camera IDs: {selected_colmap_ids}")
        images = [img for img in images if img['camera_id'] in selected_colmap_ids]
        camera_ids = selected_colmap_ids
    else:
        # 未指定 camera_ids：使用所有非 ego 相机
        images = [img for img in images if img['camera_id'] in exo_camera_ids]
        camera_ids = exo_camera_ids

    print("[INFO] 使用 {} 个相机 (ID: {})".format(len(camera_ids), camera_ids))
    print(f"[INFO] 找到 {len(images)} 张图片")

    camera_names = []
    poses = []
    intrinsics = []
    focals = []

    # 为每个相机选择一张代表图片
    camera_representative = {}
    for img in images:
        cam_id = img['camera_id']
        if cam_id not in camera_representative or img['id'] < camera_representative[cam_id]['id']:
            camera_representative[cam_id] = img

    aspect_ratio = None

    for cam_idx, cam_id in enumerate(camera_ids):
        if cam_id not in camera_representative:
            print(f"[WARNING] 相机 {cam_id} 没有图片，跳过")
            continue
        if cam_id not in cameras:
            print(f"[WARNING] 相机 {cam_id} 没有内参，跳过")
            continue

        img = camera_representative[cam_id]
        cam = cameras[cam_id]

        # 相机名称：使用索引而不是从 images.txt 读取的 name，避免打乱顺序
        if use_camera_index_as_name:
            camera_name = f"cam{(cam_idx+1):02d}"
        else:
            camera_name = f"cam{cam_id:02d}"
        camera_names.append(camera_name)

        # 图像尺寸
        img_width = cam['width']
        img_height = cam['height']

        if aspect_ratio is None:
            aspect_ratio = img_height / img_width

        # 缩放内参
        x_scale = 512 / img_width
        y_scale = 288 / img_height
        fx = cam['params'][0] * x_scale
        fy = cam['params'][1] * y_scale
        cx_scaled = 256
        cy_scaled = 144

        # COLMAP 的位姿是 world2cam，需要转换为 cam2world
        qx, qy, qz, qw = img['qx'], img['qy'], img['qz'], img['qw']
        tx, ty, tz = img['tx'], img['ty'], img['tz']

        # 归一化四元数
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        # 从四元数构建旋转矩阵 (world2cam)
        R_w2c = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])

        # 构建 world2cam 变换矩阵
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = [tx, ty, tz]

        # 转换为 cam2world
        T_cw = np.linalg.inv(T_w2c)

        poses.append(T_cw.tolist())

        # 内参矩阵
        K = np.array([[fx, 0.0, cx_scaled], [0.0, fy, cy_scaled], [0.0, 0.0, 1.0]])
        intrinsics.append(K.tolist())
        focals.append(fx)

    n_cameras = len(camera_names)
    if aspect_ratio is None:
        aspect_ratio = 9 / 16  # 默认 512x288 的宽高比

    image_height_scaled = int(image_size * aspect_ratio)

    output = {
        "sequence": "colmap",
        "sequence_key": "colmap",
        "target": "colmap",
        "camera_count": n_cameras,
        "camera_names": camera_names,
        "poses": poses,
        "intrinsics": intrinsics,
        "image_size": image_size,
        "image_width": image_size,
        "image_height": image_height_scaled,
        "focals": focals,
        "model_name": "COLMAP",
        "source_file": str(colmap_dir)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[INFO] 已保存到: {output_path}")
    return output


def convert_csv_to_pose_metadata(csv_path: Path, output_path: Path, image_size: int = 512) -> Dict:
    """将 CSV 文件转换为 pose_metadata.json 格式"""
    sequence_name = csv_path.parent.name
    if sequence_name.startswith('_'):
        sequence_name = sequence_name[1:]

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        cameras = list(reader)

    if not cameras:
        raise ValueError(f"CSV 文件为空: {csv_path}")

    n_cameras = len(cameras)
    img_width = int(cameras[0]['image_width'])
    img_height = int(cameras[0]['image_height'])

    print(f"[INFO] 找到 {n_cameras} 个相机")
    print(f"[INFO] 图像尺寸: {img_width}x{img_height}")

    camera_names = []
    poses = []
    intrinsics = []
    focals = []

    # 跳过第一个相机（ARIA ego 相机占位符），只读取后面 4 个
    for cam_data in cameras:
        cam_uid = cam_data['cam_uid']
        cam_num = int(cam_uid.replace('cam', ''))
        # if cam_num == 0:
        #     continue  # 跳过第一个相机
        camera_name = f"{sequence_name}_undist_cam{cam_num:02d}"
        camera_names.append(camera_name)

        tx = float(cam_data['tx_world_cam'])
        ty = float(cam_data['ty_world_cam'])
        tz = float(cam_data['tz_world_cam'])
        qx = float(cam_data['qx_world_cam'])
        qy = float(cam_data['qy_world_cam'])
        qz = float(cam_data['qz_world_cam'])
        qw = float(cam_data['qw_world_cam'])

        fx = float(cam_data['intrinsics_0'])
        fy = float(cam_data['intrinsics_1'])
        cx = float(cam_data['intrinsics_2'])
        cy = float(cam_data['intrinsics_3'])

        scale = image_size / max(img_width, img_height)
        x_scale = 512 / img_width
        y_scale = 288 / img_height
        fx_scaled = fx * x_scale
        fy_scaled = fy * y_scale
        cx_scaled = cx * x_scale
        cy_scaled = cy * y_scale

        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        x, y, z, w = qx/norm, qy/norm, qz/norm, qw/norm
        R_w = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])

        T_w = np.eye(4)
        T_w[:3, :3] = R_w
        T_w[:3, 3] = [tx, ty, tz]

        poses.append(T_w.tolist())

        K = np.array([[fx_scaled, 0.0, cx_scaled], [0.0, fy_scaled, cy_scaled], [0.0, 0.0, 1.0]])
        intrinsics.append(K.tolist())
        focals.append(fx_scaled)

    aspect_ratio = img_height / img_width
    image_height_scaled = int(image_size * aspect_ratio)

    output = {
        "sequence": f"_{sequence_name}",
        "sequence_key": sequence_name,
        "target": sequence_name,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[INFO] 已保存到: {output_path}")
    return output


def convert_train_meta_to_pose_metadata(train_meta_path: Path, output_path: Path, sequence_name: str) -> Dict:
    """将 train_meta.json 文件转换为 pose_metadata.json 格式"""
    with open(train_meta_path, 'r') as f:
        train_meta = json.load(f)

    # 获取数据
    hw = train_meta['hw'][0]  # [num_cameras, [h, w]]
    k = train_meta['k'][0]    # [num_cameras, 3x3 intrinsic matrix]
    w2c = train_meta['w2c'][0]  # [num_cameras, 4x4 transformation matrix] (actually c2w)

    n_cameras = len(hw)
    img_height = hw[0][1]
    img_width = hw[0][0]

    print(f"[INFO] 找到 {n_cameras} 个相机")
    print(f"[INFO] 图像尺寸: {img_width}x{img_height}")

    # 计算缩放后的图像尺寸
    image_size = 512
    aspect_ratio = img_height / img_width
    image_height_scaled = int(image_size * aspect_ratio)

    # 计算缩放比例 - 保持像素为方形，使用统一的缩放比例
    # 根据长边进行缩放，确保图像能适应目标尺寸
    if img_width >= img_height:
        # 宽度是长边
        scale = image_size / img_width
    else:
        # 高度是长边
        scale = image_height_scaled / img_height

    print(f"[INFO] 缩放后图像尺寸: {image_size}x{image_height_scaled}")
    print(f"[INFO] 统一缩放比例: scale={scale:.4f}")

    camera_names = []
    poses = []
    intrinsics = []
    focals = []

    for cam_idx in range(n_cameras):
        camera_name = f"{sequence_name}_undist_cam{cam_idx+1:02d}"
        camera_names.append(camera_name)

        # w2c 实际上是 c2w（camera to world），直接使用
        T_c2w = np.linalg.inv(np.array(w2c[cam_idx]))

        # 原始单位为 mm，需要缩小一千倍转换为 m
        T_c2w[:3, 3] = T_c2w[:3, 3]
        poses.append(T_c2w.tolist())

        # 内参矩阵
        K = np.array(k[cam_idx])

        # 获取原始的 cx, cy
        orig_cx = K[0, 2]
        orig_cy = K[1, 2]

        # 根据统一缩放比例进行放缩，保持像素为方形
        cx_scaled = orig_cx * scale
        cy_scaled = orig_cy * scale

        # 更新内参矩阵
        K[0, 2] = cx_scaled
        K[1, 2] = cy_scaled

        # 焦距也根据统一缩放比例进行缩放
        K[0, 0] = K[0, 0] * scale
        K[1, 1] = K[1, 1] * scale

        intrinsics.append(K.tolist())
        focals.append(float(K[0, 0]))

        print(f"[DEBUG] Camera {cam_idx}: orig_cx={orig_cx:.2f}, orig_cy={orig_cy:.2f} -> cx={cx_scaled:.2f}, cy={cy_scaled:.2f}")
        print(f"[DEBUG] Camera {cam_idx}: K matrix = [{K[0,0]:.2f}, {K[1,1]:.2f}, {K[0,2]:.2f}, {K[1,2]:.2f}]")

    output = {
        "sequence": f"_{sequence_name}",
        "sequence_key": sequence_name,
        "target": sequence_name,
        "camera_count": n_cameras,
        "camera_names": camera_names,
        "poses": poses,
        "intrinsics": intrinsics,
        "image_size": image_size,
        "image_width": image_size,
        "image_height": image_height_scaled,
        "focals": focals,
        "model_name": "train_meta",
        "source_file": str(train_meta_path)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[INFO] 已保存到: {output_path}")
    return output


def convert_pose_to_colmap_json(json_data: Dict, output_dir: Path, image_file_pattern: Optional[str] = None):
    """将 pose_metadata 数据转换为 COLMAP 格式"""
    camera_names = json_data['camera_names']
    poses = np.array(json_data['poses'])
    intrinsics = np.array(json_data['intrinsics'])

    n_cameras = len(camera_names)
    print(f"[INFO] 转换 {n_cameras} 个相机到 COLMAP 格式")

    from scipy.spatial.transform import Rotation
    from mast3r.colmap.read_write_model import Camera, Image, write_cameras_binary, write_images_binary

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    cameras = {}
    images = {}

    for cam_idx, camera_name in enumerate(camera_names):
        camera_id = cam_idx + 1
        image_id = cam_idx + 1

        K = intrinsics[cam_idx]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = round(float(K[0, 2]))  # 强制四舍五入：255.9999... → 256
        cy = round(float(K[1, 2]))
        print(fx, fy, cx, cy)
        cameras[camera_id] = Camera(
            id=camera_id, model='PINHOLE',
            width=int(json_data.get('image_width', 512)),
            height=int(json_data.get('image_height', 288)),
            params=np.array([fx, fy, cx, cy])
        )

        cam2world = poses[cam_idx]
        world2cam = np.linalg.inv(cam2world)
        R_w2c = world2cam[:3, :3]
        t_w2c = world2cam[:3, 3]

        q = Rotation.from_matrix(R_w2c).as_quat()
        qvec = np.array([q[3], q[0], q[1], q[2]])

        if image_file_pattern is not None:
            if 'cam' in camera_name:
                cam_num_str = camera_name.rsplit('cam', 1)[-1]
                try:
                    cam_num = int(cam_num_str)
                except ValueError:
                    cam_num = cam_idx
            else:
                cam_num = cam_idx
            img_name = image_file_pattern.format(cam_idx=cam_num-1, cam_idx_=cam_num, frame_idx=0)
            print(img_name)
        else:
            img_name = f"{camera_name}.jpg"

        images[image_id] = Image(
            id=image_id, qvec=qvec, tvec=t_w2c, camera_id=camera_id,
            name=img_name, xys=np.zeros((0, 2)),
            point3D_ids=np.zeros((0,), dtype=np.int64)
        )

    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    points_bin = sparse_dir / "points3D.bin"

    write_cameras_binary(cameras, str(cameras_bin))
    write_images_binary(images, str(images_bin))
    points_bin.write_bytes(b'')

    print(f"[INFO] 已写入 COLMAP 文件到: {sparse_dir}")
    return cameras, images


def rename_images_to_standard_format(input_dir: Path, output_dir: Path, dry_run: bool = False,
                                     start_frame: int = 1, end_frame: Optional[int] = None):
    """步骤1: 重命名图片

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        dry_run: 预览模式
        start_frame: 起始帧索引 (1-based)
        end_frame: 结束帧索引 (1-based, 不包含)，如果为 None 则到末尾
    """
    print("\n" + "="*60)
    print("[STEP 1] 重命名图片文件为标准格式")
    print("="*60)
    print(f"[INFO] 帧范围: {start_frame} 到 {end_frame-1 if end_frame else 'N'}")

    camera_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("cam")], key=natural_key)
    if not camera_dirs:
        raise FileNotFoundError(f"未找到相机目录: {input_dir}")

    print(f"[INFO] 找到 {len(camera_dirs)} 个相机目录")

    all_frames: List[List[Path]] = []
    num_frames = None
    for cam_dir in camera_dirs:
        frames = sorted([p for p in cam_dir.iterdir() if p.suffix in VALID_SUFFIXES], key=natural_key)
        if not frames:
            raise FileNotFoundError(f"相机目录 {cam_dir} 中没有图片")
        if num_frames is None:
            num_frames = len(frames)
        elif len(frames) != num_frames:
            raise ValueError(f"相机 {cam_dir.name} 有 {len(frames)} 帧")
        all_frames.append(frames)

    print(f"[INFO] 每个相机有 {num_frames} 帧")

    # 根据帧范围筛选
    if end_frame is None:
        end_frame = num_frames + 1
    selected_frame_count = end_frame - start_frame
    print(f"[INFO] 选择了 {selected_frame_count} 帧 (索引 {start_frame} 到 {end_frame-1})")

    if start_frame < 1 or start_frame > num_frames:
        raise ValueError(f"起始帧 {start_frame} 超出范围 (1-{num_frames})")
    if end_frame is not None and (end_frame > num_frames + 1 or end_frame <= start_frame):
        raise ValueError(f"结束帧 {end_frame} 超出范围或无效")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for cam_idx, cam_dir in enumerate(camera_dirs):
        cam_id = cam_idx + 1
        output_cam_dir = output_dir / f"cam{cam_id:02d}"

        if not dry_run:
            output_cam_dir.mkdir(parents=True, exist_ok=True)

        # 只处理指定范围内的帧，重命名后从1开始编号
        selected_frames = all_frames[cam_idx][start_frame-1:end_frame-1]
        for new_frame_idx, src_file in enumerate(selected_frames, start=1):
            new_name = f"cam_{cam_id:04d}_{new_frame_idx:04d}.jpg"
            dst_file = output_cam_dir / new_name

            if dry_run:
                print(f"  [DRY RUN] {src_file.name} -> {new_name}")
            else:
                shutil.copy2(src_file, dst_file)
                if (new_frame_idx - 1) % 50 == 0:
                    print(f"  [INFO] 已处理相机 {cam_id:02d} 的 {new_frame_idx}/{selected_frame_count} 帧")

    print(f"[INFO] 步骤1完成")
    return output_dir


def run_mast3r_sfm(
    input_dir: Path, output_dir: Path, sfm_config: str = "unposed",
    start_frame: int = 0, stop_frame: Optional[int] = None, skip_existing: bool = False,
    keep_working_images: bool = False, num_workers: int = 2, dry_run: bool = False,
    preset_pose_path: Optional[Path] = None, preset_pose_colmap_dir: Optional[Path] = None,
    camera_ids: Optional[List[int]] = None
):
    """步骤2: 运行 MASt3R SfM"""
    print("\n" + "="*60)
    print("[STEP 2] 运行 MASt3R SfM")
    print("="*60)

    script_path = Path(__file__).parent / "run_sfm_all.py"
    if not script_path.exists():
        raise FileNotFoundError(f"找不到 run_sfm_all.py: {script_path}")

    # Preset pose 模式处理
    if preset_pose_colmap_dir is not None:
        print("[INFO] 启用 COLMAP preset pose 模式")
        print(f"[INFO] 读取 COLMAP 目录: {preset_pose_colmap_dir}")

        frame_0_dir = output_dir / "frame_00000"
        frame_0_dir.mkdir(parents=True, exist_ok=True)

        pose_data = convert_colmap_to_pose_metadata(
            colmap_dir=preset_pose_colmap_dir,
            output_path=output_dir / "pose_metadata.json",
            camera_ids=camera_ids
        )

        convert_pose_to_colmap_json(
            pose_data, frame_0_dir,
            image_file_pattern="cam{cam_idx:02d}_cam_{cam_idx_:04d}_0001.jpg"
        )

        sfm_config = "posed"

    elif preset_pose_path is not None:
        print("[INFO] 启用 preset pose 模式")
        print(f"[INFO] 读取 pose_metadata: {preset_pose_path}")

        with open(preset_pose_path, 'r') as f:
            pose_data = json.load(f)

        frame_0_dir = output_dir / "frame_00000"
        frame_0_dir.mkdir(parents=True, exist_ok=True)

        convert_pose_to_colmap_json(
            pose_data, frame_0_dir,
            image_file_pattern="cam{cam_idx:02d}_cam_{cam_idx_:04d}_0001.jpg"
        )

        sfm_config = "posed"

    cmd = [
        sys.executable, str(script_path), str(input_dir), str(output_dir),
        "--sfm-config", sfm_config, "--start-frame", str(start_frame)
    ]
    cmd.extend(["--num-workers", str(num_workers)])

    if stop_frame is not None:
        cmd.extend(["--stop-frame", str(stop_frame)])
    if skip_existing:
        cmd.append("--skip-existing")
    if keep_working_images:
        cmd.append("--keep-working-images")

    if dry_run:
        print(f"[DRY RUN] 将执行: {' '.join(cmd)}")
        return

    print(f"[INFO] 运行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[INFO] 步骤2完成")


def preprocess_temporal_data(data_dir: Path, output_path: Optional[Path] = None, 
                            config: str = "temporal_default", n_charts: Optional[int] = None,
                            grouped_by_cams_dir: Optional[Path] = None, dry_run: bool = False, **kwargs):
    """步骤3: 预处理时序数据"""
    print("\n" + "="*60)
    print("[STEP 3] 预处理时序数据")
    print("="*60)
    
    script_path = Path(__file__).parent / "preprocess_temporal_data.py"
    if not script_path.exists():
        raise FileNotFoundError(f"找不到 preprocess_temporal_data.py: {script_path}")
    
    if output_path is None:
        output_path = data_dir / "preprocessed_temporal_data.pkl"
    
    if n_charts is None:
        if grouped_by_cams_dir is None:
            raise ValueError("必须提供 n_charts 或 grouped_by_cams_dir 参数")
        camera_dirs = [p for p in grouped_by_cams_dir.iterdir() if p.is_dir() and p.name.startswith("cam")]
        n_charts = len(camera_dirs)
        print(f"[INFO] 自动检测到 {n_charts} 个相机")
    
    cmd = [
        sys.executable, str(script_path), "-d", str(data_dir), "-o", str(output_path),
        "-c", config, "--n_charts", str(n_charts), "--batch_size", "50"
    ]
    
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    if dry_run:
        print(f"[DRY RUN] 将执行: {' '.join(cmd)}")
        return output_path
    
    print(f"[INFO] 运行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[INFO] 步骤3完成")
    return output_path


def align_charts_temporal(preprocessed_data_path: Path, output_path: Optional[Path] = None,
                         start_frame: int = 0, end_frame: Optional[int] = None, dry_run: bool = False, **kwargs):
    """步骤4: 对齐 charts"""
    print("\n" + "="*60)
    print("[STEP 4] 对齐 charts (temporal)")
    print("="*60)
    
    script_path = Path(__file__).parent / "align_charts_temporal_from_preprocessed.py"
    if not script_path.exists():
        raise FileNotFoundError(f"找不到 align_charts_temporal_from_preprocessed.py: {script_path}")
    
    if output_path is None:
        output_path = preprocessed_data_path.parent / "temporal_charts"
    
    cmd = [
        sys.executable, str(script_path), "-p", str(preprocessed_data_path), "-o", str(output_path),
        "--start_frame", str(start_frame)
    ]
    
    if end_frame is not None:
        cmd.extend(["--end-frame", str(end_frame)])
    
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    if dry_run:
        print(f"[DRY RUN] 将执行: {' '.join(cmd)}")
        return output_path
    
    print(f"[INFO] 运行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[INFO] 步骤4完成")
    return output_path


def organize_priors(input_folder: Path, priors_folder: Optional[Path] = None, dry_run: bool = False):
    """步骤5: 组织 priors"""
    print("\n" + "="*60)
    print("[STEP 5] 组织 priors 文件")
    print("="*60)
    
    script_path = Path(__file__).parent / "organize_priors.py"
    if not script_path.exists():
        raise FileNotFoundError(f"找不到 organize_priors.py: {script_path}")
    
    if priors_folder is None:
        priors_folder = input_folder.parent / "priors"
    
    cmd = [sys.executable, str(script_path), str(input_folder), "--priors-folder", str(priors_folder)]
    
    if dry_run:
        cmd.append("--dry-run")
        print(f"[DRY RUN] 将执行: {' '.join(cmd)}")
        return priors_folder
    
    print(f"[INFO] 运行命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[INFO] 步骤5完成")
    return priors_folder


def organize_final_dataset_structure(
    frame_output_dir: Path,
    preprocessed_priors_dir: Path,
    final_dataset_dir: Path,
    renamed_images_dir: Optional[Path] = None,
    dry_run: bool = False
):
    """步骤6: 整理最终数据集结构"""
    print("\n" + "="*60)
    print("[STEP 6] 整理最终数据集结构")
    print("="*60)
    
    if not dry_run:
        final_dataset_dir.mkdir(parents=True, exist_ok=True)
        mast3r_sfm_dir = final_dataset_dir / "mast3r_sfm"
        mast3r_sfm_dir.mkdir(parents=True, exist_ok=True)
    
    first_frame_dir = frame_output_dir / "frame_00000" / "mast3r_sfm"
    if not first_frame_dir.exists():
        raise FileNotFoundError(f"第一帧 mast3r_sfm 目录不存在: {first_frame_dir}")
    
    print(f"[INFO] 复制第一帧的 mast3r_sfm 数据...")
    
    items_to_copy = ["cameras.json", "sparse", "points.ply", "pointmaps"]
    
    for item in items_to_copy:
        src = first_frame_dir / item
        dst = mast3r_sfm_dir / item
        if not src.exists():
            print(f"[WARNING] 源文件/目录不存在: {src}")
            continue
        if dry_run:
            print(f"  [DRY RUN] 复制 {item}")
        else:
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            print(f"  [INFO] 已复制 {item}")
    
    print(f"[INFO] 复制相机图片...")
    
    if renamed_images_dir and renamed_images_dir.exists():
        camera_dirs = sorted([d for d in renamed_images_dir.iterdir() if d.is_dir() and d.name.startswith("cam")], key=natural_key)
        for cam_dir in camera_dirs:
            cam_match = re.match(r"cam(\d+)", cam_dir.name)
            if not cam_match:
                continue
            cam_id = int(cam_match.group(1))
            dst_cam_dir = mast3r_sfm_dir / f"cam{cam_id:02d}"
            if not dry_run:
                dst_cam_dir.mkdir(parents=True, exist_ok=True)
            img_files = sorted([f for f in cam_dir.iterdir() if f.suffix.lower() in VALID_SUFFIXES], key=natural_key)
            for img_file in img_files:
                dst_file = dst_cam_dir / img_file.name
                if dry_run:
                    print(f"  [DRY RUN] {img_file.name}")
                else:
                    shutil.copy2(img_file, dst_file)
            if not dry_run:
                print(f"  [INFO] 已复制 {len(img_files)} 张图片到 cam{cam_id:02d}/")
    
    print(f"[INFO] 复制 preprocessed_priors...")
    if preprocessed_priors_dir.exists():
        dst_priors = mast3r_sfm_dir / "preprocessed_priors"
        if dry_run:
            print(f"  [DRY RUN] 复制 preprocessed_priors")
        else:
            if dst_priors.exists():
                shutil.rmtree(dst_priors)
            shutil.copytree(preprocessed_priors_dir, dst_priors)
            print(f"  [INFO] 已复制 preprocessed_priors")
    
    print(f"[INFO] 步骤6完成")
    return final_dataset_dir


def main():
    parser = argparse.ArgumentParser(description="完整的数据集处理流程")
    parser.add_argument("input_dir", type=Path, help="输入的 grouped_by_cams 目录")
    parser.add_argument("--output_base", type=Path, default=None, help="输出基础目录")
    parser.add_argument("--skip-step", type=int, nargs="+", default=[], help="跳过的步骤")
    parser.add_argument("--only-step", type=int, nargs="+", default=None, help="只执行指定步骤")
    parser.add_argument("--dry-run", action="store_true", help="预览操作")
    parser.add_argument("--renamed-images-dir", type=Path, default=None)
    parser.add_argument("--rename-start-frame", type=int, default=1, help="重命名图片的起始帧索引 (1-based)")
    parser.add_argument("--rename-end-frame", type=int, default=None, help="重命名图片的结束帧索引 (1-based, 不包含)")
    parser.add_argument("--sfm-config", type=str, default="unposed", help="MASt3R SfM 配置")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int, default=None)
    parser.add_argument("--skip-existing-sfm", action="store_true")
    parser.add_argument("--keep-working-images", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    # Preset pose 模式
    parser.add_argument("--preset-pose-csv", type=Path, default=None, help="CSV 相机参数文件（启用 preset pose 模式）")
    parser.add_argument("--preset-pose-colmap-dir", type=Path, default=None, help="COLMAP 目录，包含 cameras.txt 和 images.txt（启用 preset pose 模式）")
    parser.add_argument("--camera-ids", type=int, nargs="+", default=None, help="指定要使用的相机ID列表")
    parser.add_argument("--train-meta-json", type=Path, default=None, help="train_meta.json 文件（启用 preset pose 模式）")
    parser.add_argument("--sequence-name", type=str, default=None, help="序列名称（配合 --train-meta-json 使用）")
    parser.add_argument("--preprocess-config", type=str, default="temporal_default")
    parser.add_argument("--preprocessed-pkl", type=Path, default=None)
    parser.add_argument("--align-start-frame", type=int, default=0)
    parser.add_argument("--align-end-frame", type=int, default=None)
    parser.add_argument("--preprocessed-depth-dir", type=Path, default=None)
    parser.add_argument("--priors-dir", type=Path, default=None)
    parser.add_argument("--final-dataset-dir", type=Path, default=None)

    args = parser.parse_args()

    if args.output_base is None:
        args.output_base = args.input_dir.parent / "dataset"
    if args.renamed_images_dir is None:
        args.renamed_images_dir = args.output_base / "renamed_images"
    frame_output_dir = args.output_base / "frames_output"
    if args.preprocessed_pkl is None:
        args.preprocessed_pkl = frame_output_dir / "preprocessed_temporal_data.pkl"
    temporal_charts_dir = frame_output_dir / "temporal_charts"
    if args.preprocessed_depth_dir is None:
        args.preprocessed_depth_dir = frame_output_dir / "preprocessed_depth"
    if args.priors_dir is None:
        args.priors_dir = args.output_base / "priors"
    if args.final_dataset_dir is None:
        args.final_dataset_dir = args.output_base / "final_dataset"
    # Presetpose 模式处理
    preset_pose_json_path = None
    preset_pose_colmap_dir = None
    if args.preset_pose_colmap_dir is not None:
        print("\n" + "="*60)
        print("[COLMAP PRESET POSE MODE] 启用 COLMAP preset pose 模式")
        print("="*60)
        preset_pose_colmap_dir = args.preset_pose_colmap_dir
    elif args.preset_pose_csv is not None:
        print("\n" + "="*60)
        print("[PRESET POSE MODE] 启用 preset pose 模式")
        print("="*60)

        # 自动确定保存路径
        sequence_name = args.preset_pose_csv.parent.name
        if sequence_name.startswith('_'):
            sequence_name = sequence_name[1:]
        save_pose_json = args.output_base / "pose_metadata.json"

        print(f"[INFO] 转换 CSV 到 JSON: {args.preset_pose_csv} -> {save_pose_json}")

        convert_csv_to_pose_metadata(
            csv_path=args.preset_pose_csv,
            output_path=save_pose_json,
            image_size=512
        )

        preset_pose_json_path = save_pose_json
    elif args.train_meta_json is not None:
        print("\n" + "="*60)
        print("[PRESET POSE MODE] 启用 preset pose 模式 (train_meta.json)")
        print("="*60)

        # 获取序列名称
        if args.sequence_name is None:
            # 尝试从 input_dir 获取
            sequence_name = args.input_dir.name
            if sequence_name.startswith('_'):
                sequence_name = sequence_name[1:]
        else:
            sequence_name = args.sequence_name

        save_pose_json = args.output_base / "pose_metadata.json"

        print(f"[INFO] 转换 train_meta.json 到 JSON: {args.train_meta_json} -> {save_pose_json}")
        print(f"[INFO] 序列名称: {sequence_name}")

        convert_train_meta_to_pose_metadata(
            train_meta_path=args.train_meta_json,
            output_path=save_pose_json,
            sequence_name=sequence_name
        )

        preset_pose_json_path = save_pose_json

    steps_to_run = []
    if args.only_step:
        steps_to_run = args.only_step
    else:
        all_steps = [1, 2, 3, 4, 5, 6]
        steps_to_run = [s for s in all_steps if s not in args.skip_step]

    print(f"\n{'='*60}")
    print(f"数据集处理流程")
    print(f"{'='*60}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出基础目录: {args.output_base}")
    print(f"将执行步骤: {steps_to_run}")
    if preset_pose_json_path is not None:
        print(f"preset pose mode: {preset_pose_json_path}")
    print(f"{'='*60}\n")

    if 1 in steps_to_run:
        renamed_dir = rename_images_to_standard_format(
            args.input_dir, args.renamed_images_dir,
            dry_run=args.dry_run,
            start_frame=args.rename_start_frame,
            end_frame=args.rename_end_frame
        )
    else:
        renamed_dir = args.renamed_images_dir

    if 2 in steps_to_run:
        run_mast3r_sfm(
            renamed_dir, frame_output_dir,
            sfm_config=args.sfm_config,
            start_frame=args.start_frame,
            stop_frame=args.stop_frame,
            skip_existing=args.skip_existing_sfm,
            keep_working_images=args.keep_working_images,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
            preset_pose_path=preset_pose_json_path,
            preset_pose_colmap_dir=preset_pose_colmap_dir,
            camera_ids=args.camera_ids
        )

    if 3 in steps_to_run:
        preprocessed_pkl = preprocess_temporal_data(
            frame_output_dir, output_path=args.preprocessed_pkl,
            config=args.preprocess_config,
            grouped_by_cams_dir=args.input_dir,
            dry_run=args.dry_run
        )
    else:
        preprocessed_pkl = args.preprocessed_pkl

    if 4 in steps_to_run:
        if not preprocessed_pkl.exists():
            raise FileNotFoundError(f"预处理数据文件不存在: {preprocessed_pkl}")
        align_charts_temporal(
            preprocessed_pkl, output_path=temporal_charts_dir,
            start_frame=args.align_start_frame,
            end_frame=args.align_end_frame,
            dry_run=args.dry_run
        )

    if 5 in steps_to_run:
        preprocessed_depth_input = None
        if temporal_charts_dir.exists():
            frame_dirs = [d for d in temporal_charts_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")]
            if frame_dirs:
                preprocessed_depth_input = temporal_charts_dir
            else:
                preprocessed_depth_subdir = temporal_charts_dir / "preprocessed_depth"
                if preprocessed_depth_subdir.exists():
                    preprocessed_depth_input = preprocessed_depth_subdir
        if preprocessed_depth_input is None:
            preprocessed_depth_input = args.preprocessed_depth_dir
        if not preprocessed_depth_input or not preprocessed_depth_input.exists():
            print(f"[WARNING] 找不到 preprocessed_depth 目录，跳过步骤5")
        else:
            organize_priors(preprocessed_depth_input, priors_folder=args.priors_dir, dry_run=args.dry_run)

    if 6 in steps_to_run:
        organize_final_dataset_structure(
            frame_output_dir, preprocessed_priors_dir=args.priors_dir,
            final_dataset_dir=args.final_dataset_dir,
            renamed_images_dir=args.renamed_images_dir,
            dry_run=args.dry_run
        )

    print(f"\n{'='*60}")
    print("处理流程完成！")
    print(f"{'='*60}")
    print(f"最终数据集目录: {args.final_dataset_dir}")
    print(f"可以开始训练了！")


if __name__ == "__main__":
    main()
