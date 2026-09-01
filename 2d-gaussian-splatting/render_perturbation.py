#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from scene.cameras import Camera

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import json
from scipy.spatial.transform import Rotation, Slerp


def save_depth_jet_heatmap(depthmaps, output_dir, idx, vmin=None, vmax=None):
    """
    保存深度图为 jet heatmap

    参数:
        depthmaps: 深度图列表
        output_dir: 输出目录
        idx: 图像索引
        vmin, vmax: 归一化的最小/最大值，如果为None则自动计算
    """
    depth_raw = depthmaps[idx][0].cpu().numpy()

    # 确定归一化范围
    if vmin is None or vmax is None:
        valid_mask = np.isfinite(depth_raw) & (depth_raw > 0)
        if np.any(valid_mask):
            vmin = np.percentile(depth_raw[valid_mask], 2) if vmin is None else vmin
            vmax = np.percentile(depth_raw[valid_mask], 98) if vmax is None else vmax
        else:
            vmin, vmax = 0, 1

    # 归一化到 [0, 1]
    depth_normalized = np.clip((depth_raw - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # 应用 jet colormap
    jet_cmap = cm.get_cmap('jet')
    depth_colored = jet_cmap(depth_normalized)[:, :, :3]  # 获取 RGB，去掉 alpha

    # 转换为 uint8
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # 保存
    Image.fromarray(depth_colored, mode='RGB').save(
        os.path.join(output_dir, 'depth_jet_{0:05d}.png'.format(idx))
    )

    return vmin, vmax


def compute_depth_range(depthmaps):
    """计算所有深度图的有效范围"""
    all_depths = []
    for depth in depthmaps:
        depth_raw = depth[0].cpu().numpy()
        valid_mask = np.isfinite(depth_raw) & (depth_raw > 0)
        if np.any(valid_mask):
            all_depths.append(depth_raw[valid_mask])
    if len(all_depths) == 0:
        return 0, 1
    all_depths = np.concatenate(all_depths)
    vmin = np.percentile(all_depths, 2)
    vmax = np.percentile(all_depths, 98)
    return vmin, vmax


def rotation_matrix_x(angle_deg):
    """绕X轴旋转的旋转矩阵"""
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ])


def rotation_matrix_y(angle_deg):
    """绕Y轴旋转的旋转矩阵"""
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])


def rotation_matrix_z(angle_deg):
    """绕Z轴旋转的旋转矩阵"""
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])


def apply_rotation_around_center(R, T, angle_x, angle_y, angle_z, scene_center=None):
    """
    应用绕场景中心的旋转

    参数:
        R: 原始旋转矩阵 (3, 3)
        T: 原始平移向量 (3,)
        angle_x, angle_y, angle_z: 绕各轴的旋转角度（度）
        scene_center: 场景中心点，如果为None则使用原点

    返回:
        R_new, T_new: 新的旋转矩阵和平移向量
    """
    if scene_center is None:
        scene_center = np.zeros(3)

    # 构建旋转矩阵
    Rx = rotation_matrix_x(angle_x)
    Ry = rotation_matrix_y(angle_y)
    Rz = rotation_matrix_z(angle_z)
    R_perturb = Rz @ Ry @ Rx  # 组合旋转

    # 计算当前相机中心在世界坐标系中的位置
    # 从R和T恢复相机中心: C = -R^T @ T
    C = -R.T @ T

    # 将相机中心相对于场景中心进行旋转
    C_rel = C - scene_center
    C_rel_rotated = R_perturb @ C_rel
    C_new = C_rel_rotated + scene_center

    # 新的旋转矩阵: 先应用原始旋转，再应用扰动旋转
    R_new = R_perturb @ R

    # 计算新的平移向量: T = -R @ C
    T_new = -R_new @ C_new

    return R_new, T_new


def create_perturbed_camera(camera, R_new, T_new, angle_x=0, angle_y=0, angle_z=0):
    """
    使用预计算的扰动后姿态创建新相机

    参数:
        camera: 原始 Camera 对象（用于获取内参和图像数据）
        R_new: 新的旋转矩阵 (3, 3)
        T_new: 新的平移向量 (3,)
        angle_x, angle_y, angle_z: 扰动角度（仅用于命名）

    返回:
        新的 Camera 对象
    """
    # 创建新的相机对象，使用预计算的R_new, T_new
    new_camera = Camera(
        colmap_id=camera.colmap_id,
        R=R_new,
        T=T_new,
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=camera.original_image,
        gt_alpha_mask=camera.gt_alpha_mask if hasattr(camera, 'gt_alpha_mask') else None,
        image_name=camera.image_name + f"_perturbed_{angle_x:.1f}_{angle_y:.1f}_{angle_z:.1f}",
        uid=camera.uid,
        time=camera.time,
        trans=camera.trans if hasattr(camera, 'trans') else np.array([0.0, 0.0, 0.0]),
        scale=camera.scale if hasattr(camera, 'scale') else 1.0,
        data_device=camera.data_device if hasattr(camera, 'data_device') else "cuda"
    )

    return new_camera


def create_interpolated_camera(camera, R_new, T_new, alpha=0.5, cam_from=None, cam_to=None):
    """
    使用插值后的姿态创建新相机。

    插值相机保留原相机的内参与时间，只替换外参。
    """
    suffix = f"_interpolated_{alpha:.3f}"
    if cam_from is not None and cam_to is not None:
        suffix += f"_{cam_from}_to_{cam_to}"

    new_camera = Camera(
        colmap_id=camera.colmap_id,
        R=R_new,
        T=T_new,
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=camera.original_image,
        gt_alpha_mask=camera.gt_alpha_mask if hasattr(camera, 'gt_alpha_mask') else None,
        image_name=camera.image_name + suffix,
        uid=camera.uid,
        time=camera.time,
        trans=camera.trans if hasattr(camera, 'trans') else np.array([0.0, 0.0, 0.0]),
        scale=camera.scale if hasattr(camera, 'scale') else 1.0,
        data_device=camera.data_device if hasattr(camera, 'data_device') else "cuda"
    )

    return new_camera


def compute_scene_center(cameras):
    """
    计算场景中心点，基于所有相机的平均位置

    参数:
        cameras: 相机列表

    返回:
        scene_center: 场景中心点 (3,)
    """
    centers = []
    for cam in cameras:
        # 从 R 和 T 计算相机中心
        C = -cam.R.T @ cam.T
        centers.append(C)
    centers = np.array(centers)
    scene_center = np.mean(centers, axis=0)
    return scene_center


def load_pose_metadata(pose_metadata_path):
    """
    从 pose_metadata.json 加载相机姿态

    参数:
        pose_metadata_path: JSON 文件路径

    返回:
        metadata: 包含相机名称、姿态、内参等信息的字典
    """
    with open(pose_metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def create_camera_from_pose_metadata(camera_name, pose_4x4, intrinsic_3x3, image_width, image_height,
                                      reference_camera, uid=0, time=0.0):
    """
    从 pose_metadata 创建 Camera 对象

    参数:
        camera_name: 相机名称
        pose_4x4: 4x4 cam2world 矩阵
        intrinsic_3x3: 3x3 内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        image_width: 图像宽度
        image_height: 图像高度
        reference_camera: 参考相机对象（用于获取图像数据和部分参数）
        uid: 唯一ID
        time: 时间参数

    返回:
        Camera 对象
    """
    pose_4x4 = np.array(pose_4x4)
    intrinsic_3x3 = np.array(intrinsic_3x3)

    # cam2world -> world2cam
    world2cam = np.linalg.inv(pose_4x4)
    R = world2cam[:3, :3]
    T = world2cam[:3, 3]

    # 从内参计算 FoV
    fx = intrinsic_3x3[0, 0]
    fy = intrinsic_3x3[1, 1]

    # FoVx = 2 * arctan(width / (2 * fx))
    FoVx = 2 * np.arctan(image_width / (2 * fx))
    FoVy = 2 * np.arctan(image_height / (2 * fy))

    # 创建新相机对象
    new_camera = Camera(
        colmap_id=reference_camera.colmap_id if reference_camera else uid,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        image=reference_camera.original_image if reference_camera else torch.ones((3, image_height, image_width)),
        gt_alpha_mask=reference_camera.gt_alpha_mask if reference_camera and hasattr(reference_camera, 'gt_alpha_mask') else None,
        image_name=camera_name,
        uid=uid,
        time=time,
        trans=reference_camera.trans if reference_camera and hasattr(reference_camera, 'trans') else np.array([0.0, 0.0, 0.0]),
        scale=reference_camera.scale if reference_camera and hasattr(reference_camera, 'scale') else 1.0,
        data_device=reference_camera.data_device if reference_camera and hasattr(reference_camera, 'data_device') else "cuda"
    )

    return new_camera


def apply_pose_metadata_to_cameras(metadata, cameras_to_modify):
    """
    将 pose_metadata 中的姿态应用到现有相机列表，保持列表长度不变

    参数:
        metadata: pose_metadata 字典
        cameras_to_modify: 要修改的相机列表（会被原地修改）

    返回:
        modified_cameras: 修改后的相机列表（同一对象）
    """
    camera_names = metadata['camera_names']
    poses = metadata['poses']  # [N, 4, 4] cam2world
    intrinsics = metadata['intrinsics']  # [N, 3, 3]
    image_width = metadata.get('image_width', metadata.get('image_size', 512))
    image_height = metadata.get('image_height', 288)  # 默认 4:3 比例

    # 构建 pose_metadata 相机映射 (通过 camera_name 匹配)
    pose_cam_map = {}
    for i, cam_name in enumerate(camera_names):
        # 确定图像名称
        image_name = f"{cam_name}.jpg"
        pose_cam_map[cam_name] = i
        pose_cam_map[image_name] = i

    modified_count = 0
    for cam in cameras_to_modify:
        # 移除可能的 _perturbed_ 后缀用于匹配
        base_name = cam.image_name.split("_perturbed_")[0]

        # 从 base_name (cam_XXXX_YYYY) 提取 camera index
        # base_name 格式: cam_0004_0245 -> camera_idx = 4
        pose_idx = None
        try:
            parts = base_name.split("_")
            if len(parts) >= 2 and parts[0] == "cam":
                camera_idx = int(parts[1])  # 提取相机编号，如 "0004" -> 4
                # 根据 camera_idx 在 pose_metadata 中寻找实际存在的相机名。
                # 兼容诸如 cpr_undist_cam04、trajectory_undist_cam04 等不同前缀。
                camera_suffix = f"cam{camera_idx:02d}"
                matching_pose_names = [
                    pose_name for pose_name in camera_names
                    if pose_name.endswith(camera_suffix)
                ]
                if len(matching_pose_names) == 1:
                    pose_idx = pose_cam_map[matching_pose_names[0]]
                elif len(matching_pose_names) > 1:
                    print(
                        f"[WARNING] 相机 {cam.image_name} 对应多个 pose_metadata 匹配 "
                        f"{matching_pose_names}，使用第一个"
                    )
                    pose_idx = pose_cam_map[matching_pose_names[0]]
        except (ValueError, IndexError):
            pass

        # 后备匹配逻辑
        if pose_idx is None:
            if base_name in pose_cam_map:
                pose_idx = pose_cam_map[base_name]
            elif cam.image_name in pose_cam_map:
                pose_idx = pose_cam_map[cam.image_name]
            else:
                # 尝试模糊匹配
                for pose_name in camera_names:
                    if base_name in pose_name or pose_name in base_name:
                        pose_idx = pose_cam_map[pose_name]
                        break

        if pose_idx is not None:
            # 从 pose_metadata 提取姿态和内参
            pose_4x4 = np.array(poses[pose_idx])
            intrinsic_3x3 = np.array(intrinsics[pose_idx])

            # cam2world -> world2cam
            world2cam = np.linalg.inv(pose_4x4)
            R = np.transpose(world2cam[:3, :3]) 
            T = world2cam[:3, 3]

            # 从内参计算 FoV
            fx = intrinsic_3x3[0, 0]
            fy = intrinsic_3x3[1, 1]
            FoVx = 2 * np.arctan(image_width / (2 * fx))
            FoVy = 2 * np.arctan(image_height / (2 * fy))

            # 原地修改相机参数
            cam.R = R
            cam.T = T
            cam.FoVx = FoVx
            cam.FoVy = FoVy

            # 重新计算派生的张量属性（这些属性在渲染时实际使用）
            from utils.graphics_utils import getWorld2View2, getProjectionMatrix
            cam.world_view_transform = torch.tensor(getWorld2View2(R, T, cam.trans, cam.scale)).transpose(0, 1).cuda()
            cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0, 1).cuda()
            cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]

            tan_fovx = np.tan(cam.FoVx / 2.0)
            tan_fovy = np.tan(cam.FoVy / 2.0)
            cam.focal_y = cam.image_height / (2.0 * tan_fovy)
            cam.focal_x = cam.image_width / (2.0 * tan_fovx)

            modified_count += 1
        else:
            print(f"[WARNING] 未找到相机 {cam.image_name} 的 pose_metadata 匹配，保持原姿态")

    print(f"从 pose_metadata 修改了 {modified_count}/{len(cameras_to_modify)} 个相机")
    return cameras_to_modify


def get_camera_id_from_image_name(image_name):
    """
    从 image_name 中提取相机 ID。

    image_name 格式通常为: cam_XXXX_YYYY 或 cam_XXXX_YYYY_perturbed_ax_ay_az
    其中 XXXX 是相机编号，YYYY 是帧编号

    参数:
        image_name: 相机的 image_name 属性

    返回:
        camera_id: 整数类型的相机 ID
    """
    # 处理可能的 _perturbed_ 后缀
    base_name = image_name.split("_perturbed_")[0]
    parts = base_name.split("_")

    # 格式应该是 cam_XXXX_YYYY
    if len(parts) >= 2 and parts[0] == "cam":
        try:
            return int(parts[1])
        except ValueError:
            pass

    # 如果无法解析，使用 image_name 的哈希作为后备
    print(f"[WARNING] Could not parse camera ID from image_name: {image_name}, using hash fallback")
    return hash(image_name) % 10000


def get_camera_perturbation(camera_id, base_angle=5.0):
    """
    基于相机ID生成确定性的扰动参数

    每个相机都有唯一的扰动方式，确保同一个相机在不同时间步保持一致

    参数:
        camera_id: 相机的唯一标识（如从 image_name 解析的相机ID）
        base_angle: 基础扰动角度（度）

    返回:
        angle_x, angle_y, angle_z: 绕各轴的旋转角度（度）
    """
    # 使用相机ID作为种子，确保每个相机有确定的扰动
    print(camera_id)
    rng = np.random.RandomState(camera_id-1)

    # 为每个相机生成固定的偏移角度
    angle_x = rng.uniform(-base_angle, base_angle)
    angle_y = rng.uniform(-base_angle, base_angle)
    angle_z = rng.uniform(-base_angle, base_angle)

    return angle_x, angle_y, angle_z


def compute_perturbed_reference_poses(cameras, base_angle, scene_center):
    """
    为每个唯一的相机（通过image_name解析）计算参考姿态及其扰动后姿态

    参数:
        cameras: 所有相机列表
        base_angle: 基础扰动角度
        scene_center: 场景中心

    返回:
        perturbed_poses: 字典，key为camera_id（从image_name解析），value为(R_new, T_new)
    """
    # 按相机ID分组，收集每个相机的所有帧
    camera_frames = {}
    for cam in cameras:
        cam_id = get_camera_id_from_image_name(cam.image_name)
        if cam_id not in camera_frames:
            camera_frames[cam_id] = []
        camera_frames[cam_id].append(cam)

    print(f"Found {len(camera_frames)} unique cameras (from image_name)")
    for cam_id, frames in camera_frames.items():
        print(f"  Camera {cam_id}: {len(frames)} frames")

    # 为每个相机的参考帧（使用第一帧）计算扰动后的姿态
    perturbed_poses = {}
    for cam_id, frames in camera_frames.items():
        # 使用第一帧作为参考姿态
        ref_cam = frames[0]

        # 获取扰动参数（基于相机ID）
        angle_x, angle_y, angle_z = get_camera_perturbation(cam_id, base_angle=base_angle)

        # 对参考姿态应用扰动
        R_new, T_new = apply_rotation_around_center(
            ref_cam.R, ref_cam.T, angle_x, angle_y, angle_z, scene_center
        )
        perturbed_poses[cam_id] = (R_new, T_new, angle_x, angle_y, angle_z)

    return perturbed_poses


def compute_interpolated_reference_poses(cameras, alpha=0.5):
    """
    在相邻相机之间生成闭环插值位姿。

    语义与 MonoFusion/render_perturbation.py 保持一致：
    对唯一相机按 camera_id 排序后，生成
    cam0->cam1, cam1->cam2, ..., camN->cam0
    的插值相机；旋转使用 SLERP，平移使用线性插值。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    camera_frames = {}
    for cam in cameras:
        cam_id = get_camera_id_from_image_name(cam.image_name)
        camera_frames.setdefault(cam_id, []).append(cam)

    camera_ids = sorted(camera_frames.keys())
    if len(camera_ids) < 2:
        raise ValueError("Need at least 2 unique cameras for interpolation")

    print(f"Found {len(camera_ids)} unique cameras for interpolation: {camera_ids}")

    interpolated_poses = {}
    for idx, cam_id_from in enumerate(camera_ids):
        cam_id_to = camera_ids[(idx + 1) % len(camera_ids)]
        cam_from = camera_frames[cam_id_from][0]
        cam_to = camera_frames[cam_id_to][0]

        R1, T1 = cam_from.R, cam_from.T
        R2, T2 = cam_to.R, cam_to.T

        key_rots = Rotation.from_matrix(np.stack([R1, R2], axis=0))
        slerp = Slerp([0, 1], key_rots)
        R_interp = slerp([alpha]).as_matrix()[0]
        T_interp = (1 - alpha) * T1 + alpha * T2

        interpolated_poses[cam_id_from] = (
            R_interp,
            T_interp,
            alpha,
            cam_id_from,
            cam_id_to,
        )

    return interpolated_poses


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters with camera perturbation")
    model = ModelParams(parser, sentinel=True)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')

    # 添加扰动相关参数
    parser.add_argument("--perturb_angle", default=5.0, type=float, help='Camera perturbation angle in degrees (default: 5). Each camera gets a unique but fixed offset based on its ID.')
    parser.add_argument("--perturb_mode", default="deterministic", type=str, choices=["deterministic"],
                        help='Perturbation mode: deterministic (each camera has a fixed unique offset based on camera ID)')
    parser.add_argument("--output_suffix", default="perturbed", type=str, help='Suffix for output directory')
    parser.add_argument("--mode", default="perturbation", type=str, choices=["perturbation", "interpolation"],
                        help='Rendering mode: perturbation or adjacent-camera interpolation')
    parser.add_argument("--alpha", default=0.5, type=float,
                        help='Interpolation factor between adjacent cameras (0.0=first camera, 1.0=next camera)')

    # 添加 pose_metadata 相关参数
    parser.add_argument("--pose_metadata", default=None, type=str, help='Path to pose_metadata.json file. If provided, use poses from JSON instead of perturbation.')

    args = get_combined_args(parser)

    dataset, opt, iteration, pipe = model.extract(args), optimization.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 检查是否使用 pose_metadata
    use_pose_metadata = args.pose_metadata is not None and os.path.exists(args.pose_metadata)

    if use_pose_metadata:
        print("Rendering with pose_metadata " + args.pose_metadata)
        print(f"Model path: {args.model_path}")

        # 加载 pose_metadata
        metadata = load_pose_metadata(args.pose_metadata)
        print(f"Loaded {metadata.get('camera_count', len(metadata['camera_names']))} cameras from pose_metadata")

        # 获取训练相机和测试相机（保持原始列表长度）
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()

        # 将 pose_metadata 应用到训练相机和测试相机
        if len(train_cameras) > 0:
            print(f"Applying pose_metadata to {len(train_cameras)} training cameras...")
            train_cameras = apply_pose_metadata_to_cameras(metadata, train_cameras)
            print(train_cameras[0].R[0])
        if len(test_cameras) > 0:
            print(f"Applying pose_metadata to {len(test_cameras)} testing cameras...")
            test_cameras = apply_pose_metadata_to_cameras(metadata, test_cameras)

        output_suffix = args.output_suffix if args.output_suffix != "perturbed" else "pose_metadata"
        output_dir = os.path.join(args.model_path, 'train', f"{output_suffix}_{scene.loaded_iter}")
        os.makedirs(output_dir, exist_ok=True)

        gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

        # 渲染训练相机（保持原始数量）
        print(f"Rendering {len(train_cameras)} training images from pose_metadata...")
        gaussExtractor.reconstruction(train_cameras)
        gaussExtractor.export_image(output_dir)

        # 保存 jet heatmap 深度图
        print(f"Saving jet heatmap depth images...")
        vis_path = os.path.join(output_dir, "vis")
        os.makedirs(vis_path, exist_ok=True)
        vmin, vmax = compute_depth_range(gaussExtractor.depthmaps)
        print(f"Depth range for jet heatmap: [{vmin:.4f}, {vmax:.4f}]")
        for idx in range(len(gaussExtractor.depthmaps)):
            save_depth_jet_heatmap(gaussExtractor.depthmaps, vis_path, idx, vmin, vmax)

        print(f"Exported {len(train_cameras)} training images to {output_dir}")

        # 渲染测试相机（如果有）
        if len(test_cameras) > 0:
            test_output_dir = os.path.join(args.model_path, 'test', f"{output_suffix}_{scene.loaded_iter}")
            os.makedirs(test_output_dir, exist_ok=True)
            print(f"Rendering {len(test_cameras)} testing images from pose_metadata...")
            gaussExtractor.reconstruction(test_cameras)
            gaussExtractor.export_image(test_output_dir)
            print(f"Exported {len(test_cameras)} testing images to {test_output_dir}")

        # 生成视频（使用训练相机数量）
        print("Creating video from rendered images...")
        create_videos(base_dir=output_dir,
                      input_dir=output_dir,
                      out_name='render_pose_metadata',
                      num_frames=len(train_cameras))
    else:
        # 原有扰动渲染逻辑 / 相机插值逻辑
        if args.pose_metadata:
            print(f"[WARNING] Pose metadata file not found: {args.pose_metadata}, falling back to {args.mode} mode")

        all_cameras = scene.getTrainCameras() + scene.getTestCameras()

        if args.mode == "interpolation":
            print("Rendering with adjacent-camera interpolation " + args.model_path)
            print(f"Interpolation alpha: {args.alpha}")
            print("Computing interpolated reference poses for each adjacent camera pair...")
            reference_poses = compute_interpolated_reference_poses(all_cameras, args.alpha)
            print(f"Pre-computed {len(reference_poses)} interpolated reference poses")
            default_suffix = "interpolated"
        else:
            print("Rendering with camera perturbation " + args.model_path)
            print(f"Perturbation mode: deterministic, base angle: {args.perturb_angle} degrees")

            # 计算场景中心
            scene_center = compute_scene_center(all_cameras)
            print(f"Scene center: {scene_center}")

            # 预计算每个相机的扰动后参考姿态（使用第一帧作为参考）
            print("Computing perturbed reference poses for each unique camera (using first frame as reference)...")
            reference_poses = compute_perturbed_reference_poses(all_cameras, args.perturb_angle, scene_center)
            print(f"Pre-computed {len(reference_poses)} perturbed reference poses")
            default_suffix = "perturbed"

        output_suffix = args.output_suffix
        if args.mode == "interpolation" and args.output_suffix == "perturbed":
            output_suffix = default_suffix

        train_dir = os.path.join(args.model_path, 'train', f"{output_suffix}_{scene.loaded_iter}")
        test_dir = os.path.join(args.model_path, 'test', f"{output_suffix}_{scene.loaded_iter}")
        gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

        if not args.skip_train:
            print(f"export training images with {args.mode}...")
            os.makedirs(train_dir, exist_ok=True)

            train_cameras = scene.getTrainCameras()
            print(f"Generating {args.mode} poses for {len(train_cameras)} training cameras using reference poses...")

            all_render_cameras = []
            for cam in train_cameras:
                cam_id = get_camera_id_from_image_name(cam.image_name)
                if args.mode == "interpolation":
                    R_new, T_new, alpha, cam_from, cam_to = reference_poses[cam_id]
                    render_cam = create_interpolated_camera(cam, R_new, T_new, alpha, cam_from, cam_to)
                else:
                    R_new, T_new, angle_x, angle_y, angle_z = reference_poses[cam_id]
                    render_cam = create_perturbed_camera(cam, R_new, T_new, angle_x, angle_y, angle_z)
                all_render_cameras.append(render_cam)

            gaussExtractor.reconstruction(all_render_cameras)
            gaussExtractor.export_image(train_dir)

            # 保存 jet heatmap 深度图
            print(f"Saving jet heatmap depth images for training...")
            train_vis_path = os.path.join(train_dir, "vis")
            os.makedirs(train_vis_path, exist_ok=True)
            vmin, vmax = compute_depth_range(gaussExtractor.depthmaps)
            print(f"Depth range for jet heatmap: [{vmin:.4f}, {vmax:.4f}]")
            for idx in range(len(gaussExtractor.depthmaps)):
                save_depth_jet_heatmap(gaussExtractor.depthmaps, train_vis_path, idx, vmin, vmax)

            print(f"Exported {len(all_render_cameras)} {args.mode} training images to {train_dir}")


        if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
            print(f"export rendered testing images with {args.mode}...")
            os.makedirs(test_dir, exist_ok=True)

            test_cameras = scene.getTestCameras()
            print(f"Generating {args.mode} poses for {len(test_cameras)} testing cameras using reference poses...")

            all_render_cameras = []
            for cam in test_cameras:
                cam_id = get_camera_id_from_image_name(cam.image_name)
                if args.mode == "interpolation":
                    R_new, T_new, alpha, cam_from, cam_to = reference_poses[cam_id]
                    render_cam = create_interpolated_camera(cam, R_new, T_new, alpha, cam_from, cam_to)
                else:
                    R_new, T_new, angle_x, angle_y, angle_z = reference_poses[cam_id]
                    render_cam = create_perturbed_camera(cam, R_new, T_new, angle_x, angle_y, angle_z)
                all_render_cameras.append(render_cam)

            gaussExtractor.reconstruction(all_render_cameras)
            gaussExtractor.export_image(test_dir)

            # 保存 jet heatmap 深度图
            print(f"Saving jet heatmap depth images for testing...")
            test_vis_path = os.path.join(test_dir, "vis")
            os.makedirs(test_vis_path, exist_ok=True)
            vmin, vmax = compute_depth_range(gaussExtractor.depthmaps)
            print(f"Depth range for jet heatmap: [{vmin:.4f}, {vmax:.4f}]")
            for idx in range(len(gaussExtractor.depthmaps)):
                save_depth_jet_heatmap(gaussExtractor.depthmaps, test_vis_path, idx, vmin, vmax)

            print(f"Exported {len(all_render_cameras)} {args.mode} testing images to {test_dir}")

        # 在导出所有图像后，生成视频
        if not args.skip_train:
            print(f"Creating video from {args.mode} training images...")
            create_videos(base_dir=train_dir,
                          input_dir=train_dir,
                          out_name=f'render_{args.mode}',
                          num_frames=len(scene.getTrainCameras()))

        if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
            print(f"Creating video from {args.mode} testing images...")
            create_videos(base_dir=test_dir,
                          input_dir=test_dir,
                          out_name=f'render_{args.mode}',
                          num_frames=len(scene.getTestCameras()))
