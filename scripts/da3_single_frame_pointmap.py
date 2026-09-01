import os
import sys
from pathlib import Path
import argparse

import torch
import numpy as np
import open3d as o3d
import json
from PIL import Image

# 确保可以 import matcha 包
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from matcha.pointmap.depthanythingv3 import load_model as load_da3_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use DepthAnything V3 (DA3) to compute a point cloud from a single multi-view frame."
    )
    parser.add_argument(
        "--frame_dir",
        type=Path,
        required=True,
        help=(
            "目录包含单帧多视角数据，例如："
            " frame_00046/，内部应包含 images/（多相机图像）"
        ),
    )
    parser.add_argument(
        "--export_dir",
        type=Path,
        default=None,
        help="DA3 导出结果目录（默认：frame_dir/da3_output）",
    )
    parser.add_argument(
        "--export_format",
        type=str,
        default="npz",
        help="DA3 导出格式（npz/mini_npz/glb/gs_ply，默认 npz）",
    )
    parser.add_argument(
        "--depthanythingv3_model",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Depth Anything V3 (DA3) 模型名称（HuggingFace 路径）",
    )
    parser.add_argument(
        "--n_charts",
        type=int,
        default=None,
        help="参与 PointMap 的视角数量（默认 None = 使用全部视角）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备（cuda 或 cpu）",
    )
    parser.add_argument(
        "--cameras_json",
        type=Path,
        default=None,
        help="相机参数 JSON 文件路径（包含 cams2world 和 focals）",
    )
    parser.add_argument(
        "--view_list",
        type=str,
        default=None,
        nargs='+',
        help="要保存的视角索引列表（例如：0 1 2 或 0-3 表示 0,1,2,3）。如果不指定，保存所有视角",
    )
    parser.add_argument(
        "--save_individual_ply",
        action="store_true",
        help="是否保存每个视角的单独 PLY 文件",
    )
    return parser.parse_args()


def depth_to_pointcloud(depth, intrinsic, extrinsic, image=None):
    """
    将深度图反投影为 3D 点云
    
    Args:
        depth: (H, W) 深度图（单位：米）
        intrinsic: (3, 3) 内参矩阵 (ixts)
        extrinsic: (4, 4) 外参矩阵 (exts，相机到世界)
        image: (H, W, 3) 可选，RGB 图像用于点云颜色
    
    Returns:
        points: (N, 3) 3D 点云（世界坐标系）
        colors: (N, 3) 点云颜色（如果提供了图像）
    """
    H, W = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # 生成像素坐标网格
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)
    
    # 计算相机坐标系下的 3D 点
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    
    # 组合为 (H*W, 3) 形状的点云（相机坐标系）
    points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # 过滤无效深度点（深度 <= 0 或过大）
    valid_mask = (depth > 0) & (depth < 1000)  # 假设深度单位是米
    points_cam = points_cam[valid_mask.reshape(-1)]
    
    # 转换为齐次坐标
    points_cam_homo = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
    
    # 应用外参矩阵，将点云从相机坐标系转换到世界坐标系
    # DA3 的 exts 是 camera-to-world 矩阵
    points_world_homo = (extrinsic @ points_cam_homo.T).T
    points_world = points_world_homo[:, :3]
    
    # 提取颜色（如果提供了图像）
    colors = None
    if image is not None:
        colors = image.reshape(-1, 3)[valid_mask.reshape(-1)]
        # 确保颜色值在 [0, 1] 范围内
        if colors.max() > 1.0:
            colors = colors / 255.0
    
    return points_world, colors


def load_cameras_from_json(cameras_json_path, image_paths):
    """
    从 cameras.json 文件加载相机参数
    
    Args:
        cameras_json_path: cameras.json 文件路径
        image_paths: 图像路径列表（用于获取图像尺寸）
    
    Returns:
        extrinsics: (N, 4, 4) camera-to-world 矩阵数组
        intrinsics: (N, 3, 3) 内参矩阵数组
    """
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)
    
    # 提取 cams2world（相机到世界的变换矩阵）
    cams2world = np.array(cameras_data['cams2world'], dtype=np.float32)  # (N, 4, 4)
    
    # 提取 focals（焦距）
    focals = np.array(cameras_data['focals'], dtype=np.float32)  # (N,)
    
    n_cameras = len(cams2world)
    print(f"[INFO] 从 cameras.json 加载了 {n_cameras} 个相机的参数")
    
    # 构建内参矩阵
    intrinsics = []
    for i, img_path in enumerate(image_paths):
        # 读取图像尺寸
        img = Image.open(img_path)
        W, H = img.size  # PIL 返回 (width, height)
        
        # 获取焦距
        focal = focals[i]
        
        # 构建内参矩阵
        # 假设主点在图像中心
        cx = W / 2.0
        cy = H / 2.0
        
        K = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        intrinsics.append(K)
    
    intrinsics = np.stack(intrinsics, axis=0)  # (N, 3, 3)
    extrinsics = cams2world  # (N, 4, 4) camera-to-world
    
    print(f"[INFO] 内参形状: {intrinsics.shape}")
    print(f"[INFO] 外参形状: {extrinsics.shape}")
    
    return extrinsics, intrinsics


def parse_view_list(view_list_str, n_views):
    """
    解析视角列表字符串
    
    Args:
        view_list_str: 视角列表字符串，例如 "0 1 2" 或 "0-3" 或 "0,1,2"
        n_views: 总视角数
    
    Returns:
        view_indices: 视角索引列表
    """
    if view_list_str is None:
        return list(range(n_views))
    
    view_indices = []
    for item in view_list_str:
        # 处理范围格式，例如 "0-3" 表示 0,1,2,3
        if '-' in item:
            parts = item.split('-')
            if len(parts) == 2:
                start = int(parts[0])
                end = int(parts[1])
                view_indices.extend(range(start, end + 1))
            else:
                raise ValueError(f"无效的范围格式: {item}")
        else:
            # 处理单个数字或逗号分隔的列表
            if ',' in item:
                view_indices.extend([int(x.strip()) for x in item.split(',')])
            else:
                view_indices.append(int(item))
    
    # 去重并排序
    view_indices = sorted(list(set(view_indices)))
    
    # 验证索引有效性
    invalid_indices = [idx for idx in view_indices if idx < 0 or idx >= n_views]
    if invalid_indices:
        raise ValueError(f"无效的视角索引: {invalid_indices}，有效范围: 0-{n_views-1}")
    
    return view_indices


def main():
    args = parse_args()

    frame_dir: Path = args.frame_dir
    if not frame_dir.exists():
        raise FileNotFoundError(f"frame_dir 不存在: {frame_dir}")

    images_dir = frame_dir / "images"

    if not images_dir.exists():
        raise FileNotFoundError(f"找不到多视角图像目录: {images_dir}")

    device = torch.device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 1. 加载 DA3 模型
    print(f"[INFO] 加载 DepthAnything V3 模型: {args.depthanythingv3_model}")
    da3_model = load_da3_model(
        model_name=args.depthanythingv3_model,
        device=device,
    )
    print("[INFO] 模型加载完成。")

    # 2. 收集多视角图像路径
    valid_suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    all_image_paths = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix in valid_suffixes
    ]
    if len(all_image_paths) == 0:
        raise FileNotFoundError(f"{images_dir} 中没有找到任何图片")

    # 3. 加载相机参数（如果提供了 cameras.json），并根据 filepaths 排序图像
    extrinsics_array = None
    intrinsics_array = None
    
    if args.cameras_json is not None:
        if not args.cameras_json.exists():
            raise FileNotFoundError(f"cameras.json 文件不存在: {args.cameras_json}")
        
        print(f"[INFO] 从 {args.cameras_json} 加载相机参数...")
        with open(args.cameras_json, 'r') as f:
            cameras_data = json.load(f)
        
        # 如果 cameras.json 中有 filepaths，按照该顺序排序图像
        if False:
            filepaths = cameras_data['filepaths']
            print(f"[INFO] cameras.json 包含 {len(filepaths)} 个文件路径")
            
            # 创建路径映射：文件名 -> 完整路径
            image_path_dict = {p.name: p for p in all_image_paths}
            
            # 按照 filepaths 的顺序排列图像
            image_paths = []
            for fp in filepaths:
                fp_name = Path(fp).name
                if fp_name in image_path_dict:
                    image_paths.append(image_path_dict[fp_name])
                else:
                    print(f"[WARNING] cameras.json 中的文件 {fp_name} 在 images 目录中未找到")
            
            if len(image_paths) != len(filepaths):
                print(f"[WARNING] 图像数量 ({len(image_paths)}) 与 cameras.json 中的数量 ({len(filepaths)}) 不匹配")
        else:
            # 如果没有 filepaths，按文件名排序
            image_paths = sorted(all_image_paths)
            print(f"[INFO] 按文件名排序图像: {image_paths}")
            print(f"[WARNING] cameras.json 中没有 filepaths，按文件名排序图像")
        
        # 加载相机参数
        extrinsics_array, intrinsics_array = load_cameras_from_json(
            args.cameras_json,
            image_paths
        )
        
        # DA3 API 期望 numpy 数组（不是 torch tensor），因为会调用 .copy() 方法
        # 确保是 numpy 数组
        if isinstance(extrinsics_array, torch.Tensor):
            extrinsics_array = extrinsics_array.cpu().numpy()
        if isinstance(intrinsics_array, torch.Tensor):
            intrinsics_array = intrinsics_array.cpu().numpy()
    else:
        # 如果没有提供 cameras.json，按文件名排序
        image_paths = sorted(all_image_paths)
    
    image_paths_str = [str(p) for p in image_paths]
    print(f"[INFO] 共找到 {len(image_paths_str)} 张多视角图像，用于 DA3 推理。")
    
    # 4. 调用 DA3 inference，导出 npz 格式（包含 depth, exts, ixts, image）
    if args.export_dir is None:
        export_dir = frame_dir / "da3_output"
    else:
        export_dir = args.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 调用 DA3 inference，导出格式: {args.export_format}")
    print(f"[INFO] 导出目录: {export_dir}")
    
    # 构建 inference 参数
    inference_kwargs = {
        'image': image_paths_str,
        'export_dir': str(export_dir),
        'export_format': args.export_format,
    }
    
    # 如果提供了相机参数，添加到 inference 参数中
    if extrinsics_array is not None:
        inference_kwargs['extrinsics'] = extrinsics_array
        print(f"[INFO] 使用提供的相机外参")
    
    if intrinsics_array is not None:
        inference_kwargs['intrinsics'] = intrinsics_array
        print(f"[INFO] 使用提供的相机内参")
    
    # DA3 inference 会导出 npz 文件到 export_dir
    prediction = da3_model.inference(**inference_kwargs)

    print("[INFO] DA3 推理完成。")
    
    # 4. 从 DA3 导出的 npz 文件中读取数据
    # DA3 会生成一个 npz 文件，文件名可能是 scene.npz 或其他
    npz_files = list(export_dir.glob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"在 {export_dir} 中未找到 DA3 导出的 npz 文件")
    
    # 使用第一个找到的 npz 文件（通常只有一个）
    da3_npz_path = npz_files[0]
    print(f"[INFO] 读取 DA3 导出的 npz 文件: {da3_npz_path}")
    
    da3_data = np.load(da3_npz_path)
    print(f"[INFO] NPZ 文件包含的键: {list(da3_data.keys())}")
    
    # 提取数据
    depths = da3_data['depth']  # (N, H, W)
    exts = da3_data['extrinsics']  # (N, 4, 4) camera-to-world
    ixts = da3_data['intrinsics']  # (N, 3, 3) intrinsics
    
    # image 可能不存在（mini_npz 格式）
    images = da3_data.get('image', None)  # (N, H, W, 3) 或 None
    
    # conf 可能不存在
    confs = da3_data.get('conf', None)  # (N, H, W) 或 None
    
    n_views = depths.shape[0]
    print(f"[INFO] 共 {n_views} 个视角")
    print(f"[INFO] 深度图形状: {depths.shape}")
    print(f"[INFO] 外参形状: {exts.shape}")
    print(f"[INFO] 内参形状: {ixts.shape}")
    if images is not None:
        print(f"[INFO] 图像形状: {images.shape}")
    
    # 5. 解析视角列表
    view_indices = parse_view_list(args.view_list, n_views)
    print(f"[INFO] 将处理以下视角: {view_indices}")
    
    # 6. 对指定视角进行深度图反投影
    all_points = []
    all_colors = []
    selected_depths = []
    selected_exts = []
    selected_ixts = []
    selected_images = []
    selected_confs = []
    
    print(f"[INFO] 开始反投影深度图到 3D 点云...")
    for idx in view_indices:
        depth = depths[idx]  # (H, W)
        print(f"[INFO] 视角 {idx}: 深度图形状: {depth.shape}")
        intrinsic = ixts[idx]  # (3, 3)
        extrinsic = exts[idx]  # (4, 4) camera-to-world
        
        # 获取图像（如果存在）
        image = None
        if images is not None:
            image = images[idx]  # (H, W, 3)
        
        # 反投影到 3D 点云
        points, colors = depth_to_pointcloud(
            depth=depth,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            image=image,
        )
        
        all_points.append(points)
        all_colors.append(colors)
        selected_depths.append(depth)
        selected_exts.append(extrinsic)
        selected_ixts.append(intrinsic)
        if images is not None:
            selected_images.append(image)
        if confs is not None:
            selected_confs.append(confs[idx])
        
        print(f"[INFO] 视角 {idx}/{n_views-1}: 生成 {len(points)} 个 3D 点")
        
        # 如果启用单独保存，为每个视角保存单独的 PLY 文件
        if args.save_individual_ply:
            individual_ply_path = export_dir / f"pointcloud_view_{idx:04d}.ply"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                colors_normalized = colors.copy()
                if colors_normalized.max() > 1.0:
                    colors_normalized = np.clip(colors_normalized / 255.0, 0.0, 1.0)
                pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
            o3d.io.write_point_cloud(str(individual_ply_path), pcd)
            print(f"[INFO] 已保存视角 {idx} 的单独 PLY 文件: {individual_ply_path}")
    
    # 7. 保存反投影后的点云为 npz 格式（只包含选定的视角）
    if len(view_indices) == n_views:
        output_npz_path = export_dir / "pointcloud.npz"
    else:
        view_str = "_".join([str(idx) for idx in view_indices])
        output_npz_path = export_dir / f"pointcloud_views_{view_str}.npz"
    
    print(f"[INFO] 保存反投影点云到 {output_npz_path}...")
    
    save_dict = {
        # 点云数据（每个视角的点云）
        'points': all_points,  # List of (N_i, 3) arrays
        # 深度图（只包含选定的视角）
        'depths': np.stack(selected_depths, axis=0),  # (M, H, W) where M=len(view_indices)
        # 相机参数（只包含选定的视角）
        'intrinsics': np.stack(selected_ixts, axis=0),  # (M, 3, 3)
        'extrinsics': np.stack(selected_exts, axis=0),  # (M, 4, 4) camera-to-world
        # 视角索引
        'view_indices': np.array(view_indices, dtype=np.int32),
    }
    
    # 添加颜色（如果存在）
    if all_colors[0] is not None:
        save_dict['colors'] = all_colors  # List of (N_i, 3) arrays
    
    # 添加图像（如果存在）
    if selected_images:
        save_dict['images'] = np.stack(selected_images, axis=0)  # (M, H, W, 3)
    
    # 添加置信度（如果存在）
    if selected_confs:
        save_dict['conf'] = np.stack(selected_confs, axis=0)  # (M, H, W)
    
    np.savez(str(output_npz_path), **save_dict)
    
    # 8. 保存为 PLY 格式（合并选定视角的点云）
    if len(view_indices) == n_views:
        output_ply_path = export_dir / "pointcloud.ply"
    else:
        view_str = "_".join([str(idx) for idx in view_indices])
        output_ply_path = export_dir / f"pointcloud_views_{view_str}.ply"
    
    print(f"[INFO] 保存合并的点云到 PLY 文件: {output_ply_path}...")
    
    # 合并所有视角的点云
    merged_points = np.concatenate(all_points, axis=0)
    
    # 合并颜色（如果存在）
    merged_colors = None
    if all_colors[0] is not None:
        merged_colors = np.concatenate(all_colors, axis=0)
        # 确保颜色值在 [0, 1] 范围内
        if merged_colors.max() > 1.0:
            merged_colors = np.clip(merged_colors / 255.0, 0.0, 1.0)
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    
    if merged_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    
    # 保存 PLY 文件
    o3d.io.write_point_cloud(str(output_ply_path), pcd)
    
    print("[INFO] 处理完成！")
    print(f"[INFO] DA3 原始导出位于: {export_dir}")
    print(f"[INFO] 反投影点云 NPZ 文件: {output_npz_path}")
    print(f"[INFO] 合并点云 PLY 文件: {output_ply_path}")
    print(f"[INFO] 包含 {len(all_points)} 个视角的点云数据")
    print(f"[INFO] 总点数: {len(merged_points)}")


if __name__ == "__main__":
    main()


