import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import yaml
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from matcha.pointmap.depthanythingv2 import (
    PointMapDepthAnything,
    apply_depthanything,
    export_pointmap_to_pcd,
    fit_depth_to_point_cloud,
    load_model,
    get_pointmap_from_mast3r_scene_with_depthanything,
)
from matcha.pointmap.mast3r import compute_mast3r_scene
from matcha.dm_utils.rendering import fov2focal
from matcha.dm_scene.cameras import CamerasWrapper, create_gs_cameras_from_pointmap, rescale_cameras
from matcha.dm_trainers.charts_alignment import align_charts_in_parallel

# 颜色可视化支持
import colorsys


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def natural_key(path: Path) -> List[object]:
    """Return a key list so that Path objects are sorted in natural order."""
    import re
    print([int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.stem)]
)
    return [int(text) if text.isdigit() else None for text in re.split(r"(\d+)", path.stem)]


def collect_multicam_frames(root: Path) -> Tuple[List[Path], List[List[Path]]]:
    """Collect per-camera image sequences.

    Args:
        root: Root directory containing one subdirectory per camera.

    Returns:
        A tuple with the sorted list of camera directories and the aligned frame paths per camera.

    Raises:
        FileNotFoundError: If no camera directories or images are found.
        ValueError: If the cameras do not share the same number of frames.
    """

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    camera_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not camera_dirs:
        raise FileNotFoundError(f"No camera subdirectories found in {root}")
    #print(camera_dirs)
    all_frames: List[List[Path]] = []
    num_frames = None
    camera_dirs = sorted(
            [p for p in camera_dirs],
            key=natural_key,
        )
    for cam_dir in camera_dirs:
        frames = sorted(
            [p for p in cam_dir.iterdir() if p.suffix in VALID_SUFFIXES],
            key=natural_key,
        )
        #print(frames)
        #(frames)
        if not frames:
            raise FileNotFoundError(f"Camera directory {cam_dir} contains no images")
        if num_frames is None:
            num_frames = len(frames)
        elif len(frames) != num_frames:
            raise ValueError(
                f"Camera {cam_dir.name} has {len(frames)} frames, expected {num_frames}"
            )
        all_frames.append(frames)

    return camera_dirs, all_frames


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def run_mast3r_sfm(images_dir: Path, output_dir: Path, config: str, verbose: bool = True, not_first_frame: bool = True):
    ensure_dir(output_dir)
    #print(not_first_frame)
    cmd = [
        "python",
        "scripts/run_sfm_multiframes.py",
        "-s",
        str(images_dir),
        "-o",
        str(output_dir),
        "-c",
        config,
        "--not_first_frame",
        str(not_first_frame),
    ]
    if verbose:
        print("[INFO] Running MASt3R SfM:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent.parent)


def build_aligned_pointmap_with_depthanything(
    mast3r_output_dir: Path,
    image_dir: Path,
    depthanything_model,
    device: torch.device,
    align_config: Dict = None,
    charts_output_dir=None,
    depth_output_dir=None,
    conf_output_dir=None,
) -> PointMapDepthAnything:
    """Build pointmap with proper depth alignment using ParallelAligner."""
    
    if align_config is None:
        # 默认对齐配置
        align_config = {
            'use_learnable_confidence': True,
            'use_normal_loss': True,
            'use_curvature_loss': True,
            'use_matching_loss': True,
            'n_iterations': 600,
            'encodings_lr': 1e-2,
            'mlp_lr': 1e-3,
            'confidence_lr': 1e-3,
            'normal_loss_weight': 4.0,
            'curvature_loss_weight': 1.0,
            'matching_loss_weight': 5.0,
            'matching_thr_factor': 1./20.,
            'verbose': True,
        }
    
    # 第一步：使用get_pointmap_from_mast3r_scene_with_depthanything获取初始点云
    print("[INFO] Step 1: Building initial pointmap from MASt3R scene with DepthAnything...")
    config_path = os.path.join('configs/charts_alignment', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pm_config = config['pointmap']
    scene_config = config['scene']
    align_config = config['alignment']
    masking_config = config['masking']
    scene_pm, sfm_data, mast3r_pm = get_pointmap_from_mast3r_scene_with_depthanything(
        scene_source_path=str(image_dir.parent),  # 使用图像目录的父目录作为场景源
        n_images_in_pointmap=None,  # 使用所有图像
        image_indices=None,
        white_background=False,
        # MASt3R
        mast3r_scene_source_path=str(mast3r_output_dir),
        # DepthAnything
        depthanything_checkpoint_dir='./Depth-Anything-V2/checkpoints/',
        depthanything_encoder='vitl',
        # Misc
        device=device,
        return_sfm_data=True,
        return_mast3r_pointmap=True,
        **pm_config,
    )
    print("[INFO] Saving prior point cloud data...")
    save_path = mast3r_output_dir / Path("output_pointcloud.ply")
    export_pointmap_to_pcd(scene_pm, save_path=str(save_path))
    print(f"[INFO] Point cloud saved to: {save_path}")
    # 第二步：构建相机系统和对齐参数（参考align_charts.py）
    print("[INFO] Step 2: Preparing cameras and reference data for alignment...")
    
    # 构建相机包装器
    cam_list = create_gs_cameras_from_pointmap(
        scene_pm,
        image_resolution=1,
        load_gt_images=True,
        max_img_size=1600,
        use_original_image_size=True,
        average_focal_distances=False,
        verbose=False,
    )
    print(len(cam_list))
    pointmap_cameras = CamerasWrapper(cam_list, no_p3d_cameras=False)
    
    # 计算空间范围并重新缩放
    target_scale = 5.0  # 默认目标尺度
    scale_factor = target_scale / pointmap_cameras.get_spatial_extent()
    pointmap_cameras = rescale_cameras(pointmap_cameras, scale_factor)
    
    # 第三步：计算参考数据（将SfM点转换到相机坐标系）
    print("[INFO] Step 3: Computing reference data from SfM points...")
    
    reference_data = torch.cat([
        pointmap_cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
            scale_factor * sfm_data['sfm_xyz'][sfm_data['image_sfm_points'][pointmap_cameras.gs_cameras[i_chart].image_name.split('.')[0]]]
        )[..., 2].view(scene_pm.points3d[i_chart][..., 0].shape)[None]
        for i_chart in range(len(pointmap_cameras))
    ], dim=0)
    print('ddddddddd',reference_data.max())
    # 点云
    # 创建掩码（可选）
    masks = None
    if True:  # 可以添加掩码逻辑
        # mast3r_masks = mast3r_pm.confidence > some_threshold
        # masks = mast3r_masks
        pass
    
    # 第四步：执行深度图对齐优化（关键步骤！）
    print("[INFO] Step 4: Aligning depth maps using ParallelAligner...")
    #print(reference_data.shape)
    output = align_charts_in_parallel(
        # Scene
        scene_pm,
        # Data parameters
        reference_data,
        masks=masks,
        rendering_size=1600,
        target_scale=target_scale,
        verbose=True,
        return_training_losses=True,
        reprojection_matches_file=None,
        save_charts_data=True, 
        charts_data_path=charts_output_dir,# 不保存中间数据
        **align_config,
    )
    
    # 第五步：处理对齐结果
    print("[INFO] Step 5: Processing alignment results...")
    
    if align_config['use_learnable_confidence']:
        output_verts, output_depths, output_confs, training_losses = output
        output_confs = output_confs - 1.0  # 调整置信度
        print(output_depths.min(), output_depths.max())
        print(output_confs.min(), output_confs.max())
    else:
        output_verts, output_depths, training_losses = output
    
    print(f"[INFO] Alignment complete!")
    print(f"[INFO] Output vertices shape: {output_verts.shape}")
    print(f"[INFO] Output depths shape: {output_depths.shape}")
    if align_config['use_learnable_confidence']:
        print(f"[INFO] Output confidence shape: {output_confs.shape}")
    
    # 保存对齐后的depths和confs到对应目录
    if depth_output_dir is not None or conf_output_dir is not None:
        print("[INFO] Saving aligned depths and confidences...")
        
        # 确保目录存在
        if depth_output_dir is not None:
            depths_subdir = Path(depth_output_dir) 
            ensure_dir(depths_subdir)
        
        if conf_output_dir is not None and align_config['use_learnable_confidence']:
            confs_subdir = Path(conf_output_dir) 
            ensure_dir(confs_subdir)
        
        # 遍历每个相机，保存对应的深度和置信度
        for i in range(len(pointmap_cameras)):
            # 从相机图像名称解析帧ID
            image_name = pointmap_cameras.gs_cameras[i].image_name
            frame_match = image_name.split('_')
            
            try:
                if len(frame_match) >= 3:
                    # 格式: camXX_frame_XXXXX.jpg -> 提取XXXXX
                    frame_id_str = frame_match[2].split('.')[0]
                    frame_id = int(frame_id_str)
                elif len(frame_match) >= 2:
                    # 备用格式
                    frame_id_str = frame_match[1].split('.')[0]
                    frame_id = int(frame_id_str)
                else:
                    frame_id = i
            except:
                frame_id = i
            
            # 相机ID映射: 相机索引0-9 -> cam_id=0, 10-19 -> cam_id=1, 等等
            cam_id = i
            frame_id = frame_id - 1  # 转换为0-based索引
            
            # 构建文件名
            depth_file_name = f"cam{cam_id:02d}_cam_{cam_id:04d}_{frame_id:04d}_depth.npy"
            conf_file_name = f"cam{cam_id:02d}_cam_{cam_id:04d}_{frame_id:04d}_conf.npy"
            # 保存深度
            if depth_output_dir is not None:
                # output_depths可能是张量或列表
                if isinstance(output_depths, torch.Tensor):
                    depth = output_depths[i].detach().cpu().numpy()
                else:
                    depth = output_depths[i].detach().cpu().numpy() if hasattr(output_depths[i], 'detach') else np.array(output_depths[i])
                
                # 确保是2D数组 (H, W) 或 (1, H, W)
                if depth.ndim == 3 and depth.shape[0] == 1:
                    depth = depth[0]
                
                depth_path = depths_subdir / depth_file_name
                np.save(depth_path, depth.astype(np.float32))
            
            # 保存置信度
            if conf_output_dir is not None and align_config['use_learnable_confidence']:
                # output_confs可能是张量或列表
                if isinstance(output_confs, torch.Tensor):
                    conf = output_confs[i].detach().cpu().numpy()
                else:
                    conf = output_confs[i].detach().cpu().numpy() if hasattr(output_confs[i], 'detach') else np.array(output_confs[i])
                
                # 确保是2D数组 (H, W) 或 (1, H, W)
                if conf.ndim == 3 and conf.shape[0] == 1:
                    conf = conf[0]
                
                conf_path = confs_subdir / conf_file_name
                np.save(conf_path, conf.astype(np.float32))
        
        print(f"[INFO] Successfully saved depths and confidences for {len(pointmap_cameras)} cameras")
    
    # 第六步：创建最终的对齐点云
    print("[INFO] Step 6: Creating final aligned pointmap...")
    
    # 使用对齐后的数据创建新的pointmap
    aligned_pointmap = PointMapDepthAnything(
        scene_cameras=scene_pm.scene_cameras,
        scene_eval_cameras=scene_pm.scene_eval_cameras,
        img_paths=scene_pm.img_paths,
        images=scene_pm.images,
        original_images=scene_pm.original_images,
        focals=scene_pm.focals,
        poses=scene_pm.poses,
        points3d=[verts.view(scene_pm.points3d[i].shape) for i, verts in enumerate(output_verts)],
        confidence=output_confs if align_config['use_learnable_confidence'] else [torch.ones_like(depth) for depth in output_depths],
        masks=scene_pm.masks,
        device=device,
    )
    
    # 第七步：保存点云数据
    
    
    return aligned_pointmap

def depth_from_pointmap(pointmap: PointMapDepthAnything, cam_idx: int) -> np.ndarray:
    pts = pointmap.points3d[cam_idx]
    h, w = pts.shape[:2]
    pts_flat = pts.reshape(-1, 3)
    camera = pointmap.scene_cameras.p3d_cameras[cam_idx].to(pts_flat.device)
    cam_points = camera.get_world_to_view_transform().transform_points(pts_flat).view(h, w, 3)
    return cam_points[..., 2].detach().cpu().numpy()


def save_depths(pointmap: PointMapDepthAnything, output_dir: Path):
    ensure_dir(output_dir)
    for cam_idx, img_path in enumerate(pointmap.img_paths):
        depth = depth_from_pointmap(pointmap, cam_idx)
        name = Path(img_path).stem
        np.save(output_dir / f"{name}_depth.npy", depth.astype(np.float32))


def save_confidence(pointmap: PointMapDepthAnything, output_dir: Path):
    ensure_dir(output_dir)
    for cam_idx, img_path in enumerate(pointmap.img_paths):
        conf = pointmap.confidence[cam_idx].cpu().numpy()
        name = Path(img_path).stem
        np.save(output_dir / f"{name}_conf.npy", conf.astype(np.float32))


def generate_camera_colors(n_cameras, color_palette='distinct', device='cuda'):
    """
    为每个相机生成区分颜色
    
    参数:
        n_cameras: 相机数量
        color_palette: 颜色方案
        device: 计算设备
    
    返回:
        colors: (n_cameras, 3) RGB颜色值，范围[0, 1]
    """
    if color_palette == 'distinct':
        # 使用区分度高的颜色
        distinct_colors = [
            [1.0, 0.0, 0.0],  # 红色
            [0.0, 1.0, 0.0],  # 绿色
            [0.0, 0.0, 1.0],  # 蓝色
            [1.0, 1.0, 0.0],  # 黄色
            [1.0, 0.0, 1.0],  # 紫色
            [0.0, 1.0, 1.0],  # 青色
            [1.0, 0.5, 0.0],  # 橙色
            [0.5, 0.0, 1.0],  # 紫罗兰
            [0.0, 0.5, 1.0],  # 天蓝
            [0.5, 1.0, 0.0],  # 黄绿
            [1.0, 0.0, 0.5],  # 玫瑰红
            [0.0, 1.0, 0.5],  # 春绿
        ]
        
        colors = []
        for i in range(n_cameras):
            color_idx = i % len(distinct_colors)
            colors.append(distinct_colors[color_idx])
            
    elif color_palette == 'rainbow':
        # 彩虹色渐变
        colors = []
        for i in range(n_cameras):
            hue = i / max(1, n_cameras - 1)  # 0到1
            # HSV到RGB转换（饱和度=1，亮度=1）
            c = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append(list(c))
            
    elif color_palette == 'heatmap':
        # 热力图颜色（从蓝到红）
        colors = []
        for i in range(n_cameras):
            t = i / max(1, n_cameras - 1)  # 0到1
            # 蓝色(冷)到红色(热)的渐变
            r = t
            g = 0.0
            b = 1.0 - t
            colors.append([r, g, b])
            
    else:
        # 默认使用随机颜色
        torch.manual_seed(42)  # 固定种子以确保可重复性
        colors = torch.rand(n_cameras, 3).tolist()
    
    return torch.tensor(colors, device=device, dtype=torch.float32)


def save_colored_pointcloud(pointmap: PointMapDepthAnything, output_dir: Path, color_palette: str = "distinct"):
    """保存带颜色的点云数据，不同相机用不同颜色"""
    ensure_dir(output_dir)
    
    # 生成相机颜色
    n_cameras = len(pointmap.points3d)
    colors = generate_camera_colors(n_cameras, color_palette, pointmap.points3d.device)
    
    # 收集所有点云和对应的颜色
    all_points = []
    all_colors = []
    
    for cam_idx, img_path in enumerate(pointmap.img_paths):
        # 获取当前相机的点云
        pts = pointmap.points3d[cam_idx]  # (H, W, 3)
        h, w = pts.shape[:2]
        
        # 重塑为点列表
        pts_flat = pts.reshape(-1, 3)
        
        # 为当前相机的所有点分配相同颜色
        cam_color = colors[cam_idx].view(1, 3).expand(pts_flat.shape[0], -1)
        
        all_points.append(pts_flat)
        all_colors.append(cam_color)
    
    # 合并所有点云和颜色
    all_points = torch.cat(all_points, dim=0)
    all_colors = torch.cat(all_colors, dim=0)
    
    # 保存为NPZ格式，包含点和颜色信息
    output_path = output_dir / "colored_pointcloud.npz"
    np.savez(
        output_path,
        points=all_points.cpu().numpy(),
        colors=all_colors.cpu().numpy(),
        camera_colors=colors.cpu().numpy(),
        n_cameras=n_cameras,
        color_palette=color_palette
    )
    
    print(f"[INFO] 带颜色的点云已保存到: {output_path}")
    print(f"[INFO] 总点数: {all_points.shape[0]}")
    print(f"[INFO] 相机数量: {n_cameras}")
    print(f"[INFO] 颜色方案: {color_palette}")
    
    # 显示颜色映射信息
    for i in range(n_cameras):
        color = colors[i]
        print(f"  相机 {i}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
    
    
    
    # 如果可能，也保存PLY格式
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(all_colors.cpu().numpy())
        
        ply_path = output_dir / "colored_pointcloud.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"[INFO] PLY格式点云已保存到: {ply_path}")
        
    except ImportError:
        print("[INFO] 未安装open3d，无法保存PLY格式点云")
    except Exception as e:
        print(f"[WARNING] 保存PLY文件时出错: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-frame multi-camera SfM followed by DepthAnything alignment"
    )
    parser.add_argument("dataset_root", type=Path, help="Root folder containing camera subdirectories")
    parser.add_argument("output_root", type=Path, help="Directory to store intermediate and output data")
    parser.add_argument(
        "--depthanything-checkpoints",
        type=Path,
        default=Path("./Depth-Anything-V2/checkpoints"),
        help="Directory with DepthAnything checkpoints",
    )
    parser.add_argument(
        "--depthanything-encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sfm-config", type=str, default="unposed")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int, default=None, help="Exclusive upper bound; defaults to all frames")
    parser.add_argument("--skip-existing", action="store_true", help="Skip frames with existing depth outputs")
    parser.add_argument("--keep-working-images", action="store_true", help="Keep per-frame image copies")
    parser.add_argument("--save-pointcloud", action="store_true", help="Export per-frame aligned point cloud")
    parser.add_argument("--visualize", action="store_true", help="Enable colored visualization for different cameras")
    parser.add_argument("--color-palette", type=str, default="distinct", choices=["distinct", "rainbow", "heatmap"], help="Color palette for camera visualization")
    
    # 对齐参数
    parser.add_argument("--align-depths", action="store_true", help="Use ParallelAligner to align depth maps")
    parser.add_argument("--alignment-iterations", type=int, default=600, help="Number of alignment iterations")
    parser.add_argument("--use-confidence", action="store_true", help="Use learnable confidence during alignment")
    return parser.parse_args()


def main():
    args = parse_args()

    # camera_dirs, all_frames = collect_multicam_frames(args.dataset_root)
    # num_frames = len(all_frames[0])
    frame_dirs, all_frames = collect_multicam_frames(args.dataset_root)
    #print(all_frames)
    num_cameras = len(all_frames[1])
    print("num cameras:", num_cameras)
    #print(num_cameras)
    start = max(args.start_frame, 0)
    stop = len(frame_dirs) if args.stop_frame is None else min(args.stop_frame, len(frame_dirs))
    if start >= stop:
        raise ValueError("Invalid frame range")

    ensure_dir(args.output_root)

    device = torch.device(args.device)
    depthanything_model = load_model(
        checkpoint_dir=str(args.depthanything_checkpoints),
        encoder=args.depthanything_encoder,
        device=device,
    )
    mast3r_frame0_scene = args.output_root / f"frame_00000" / "mast3r_sfm"
    for frame_idx in range(start, stop):
        frame_name = f"frame_{frame_idx:05d}"
        frame_root = args.output_root / frame_name
        images_dir = frame_root / "images"
        mast3r_output_dir = frame_root / "mast3r_sfm"
        depth_output_dir = frame_root / "depth"
        conf_output_dir = frame_root / "confidence"
        charts_output_dir = frame_root / "charts"
        ensure_dir(charts_output_dir)
        print(f"[INFO] Processing {frame_name} ({frame_idx + 1 - start}/{stop - start})")

        if args.skip_existing and depth_output_dir.exists() and any(depth_output_dir.glob("*_depth.npy")):
            print(f"[INFO] Skipping {frame_name} (depth already exists)")
            continue

        ensure_dir(images_dir)

        for cam_idx in range(num_cameras):
            src = all_frames[frame_idx][cam_idx]
            print("src:", src)
            dst = images_dir / f"cam{cam_idx:02d}_{src.name}"
            symlink_or_copy(src, dst)
            
        
        not_first_frame = False if frame_idx == 0 else True
        #print(not_first_frame)
        if not (mast3r_output_dir / "cameras.json").exists():
           
            run_mast3r_sfm(images_dir, mast3r_output_dir, args.sfm_config, not_first_frame=not_first_frame)
        else:
            print(f"[INFO] Reusing existing MASt3R output for {frame_name}")

        sfm_data = compute_mast3r_scene(
            mast3r_scene_source_path=str(mast3r_output_dir),
            n_images_in_pointmap=num_cameras,
            device=str(device),
        )

        if args.align_depths:
            # 使用ParallelAligner进行深度图对齐
            print(f"[INFO] Using ParallelAligner for depth alignment...")
            align_config = {
                'use_learnable_confidence': args.use_confidence,
                'use_normal_loss': True,
                'use_curvature_loss': True,
                'use_matching_loss': True,
                'n_iterations': args.alignment_iterations,
                'encodings_lr': 1e-2,
                'mlp_lr': 1e-3,
                'confidence_lr': 1e-3,
                'normal_loss_weight': 4.0,
                'curvature_loss_weight': 1.0,
                'matching_loss_weight': 5.0,
                'matching_thr_factor': 1./20.,
                'verbose': True,
                'charts_data_path': charts_output_dir,
            }
            
            pointmap = build_aligned_pointmap_with_depthanything(
                mast3r_output_dir=mast3r_output_dir,
                image_dir=images_dir,
                depthanything_model=depthanything_model,
                device=device,
                align_config=align_config,
                charts_output_dir=charts_output_dir,
                depth_output_dir=depth_output_dir,
                conf_output_dir=conf_output_dir,
                
            )
        else:
            # 使用原始方法（不对齐）
            pointmap = build_pointmap_with_depthanything(
                sfm_data=sfm_data,
                image_dir=images_dir,
                depthanything_model=depthanything_model,
                device=device,
            )

        

        if args.save_pointcloud:
            export_pointmap_to_pcd(pointmap, save_path=str(frame_root / "aligned_pointmap.ply"))

        # 保存带颜色的点云可视化
        if args.visualize:
            print(f"[INFO] 生成带颜色的点云可视化...")
            colored_output_dir = frame_root / "colored_visualization"
            save_colored_pointcloud(pointmap, colored_output_dir, args.color_palette)

        if not args.keep_working_images:
            shutil.rmtree(images_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
