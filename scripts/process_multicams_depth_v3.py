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

from matcha.pointmap.depthanythingv3 import (
    PointMapDepthAnything,
    export_pointmap_to_pcd,
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


def build_aligned_pointmap_with_depthanything_v3(
    mast3r_output_dir: Path,
    image_dir: Path,
    depthanything_model,
    device: torch.device,
    align_config: Dict = None,
    charts_output_dir=None,
    depth_output_dir=None,
    conf_output_dir=None,
) -> PointMapDepthAnything:
    """Build pointmap with proper depth alignment using Depth Anything V3."""

    if align_config is None:
        # Default alignment config
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

    # Step 1: Build initial pointmap with Depth Anything V3
    print("[INFO] Step 1: Building initial pointmap from MASt3R scene with DepthAnything V3...")
    config_path = os.path.join('configs/charts_alignment', 'default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pm_config = config['pointmap']
    scene_config = config['scene']
    align_config_from_file = config['alignment']
    masking_config = config['masking']

    # Merge align_config
    align_config_from_file.update(align_config)
    align_config = align_config_from_file

    scene_pm, sfm_data = get_pointmap_from_mast3r_scene_with_depthanything(
        scene_source_path=str(image_dir.parent),
        mast3r_scene_source_path=str(mast3r_output_dir),
        n_images_in_pointmap=None,
        image_indices=None,
        white_background=False,
        depthanything_model_name=depthanything_model.__class__.__name__,  # Pass model directly in practice
        device=str(device),
        return_sfm_data=True,
        return_mast3r_pointmap=False,
        **pm_config,
    )

    print("[INFO] Saving prior point cloud data...")
    save_path = mast3r_output_dir / Path("output_pointcloud.ply")
    export_pointmap_to_pcd(scene_pm, save_path=str(save_path))
    print(f"[INFO] Point cloud saved to: {save_path}")

    # Step 2: Build camera system
    print("[INFO] Step 2: Preparing cameras and reference data for alignment...")

    cam_list = create_gs_cameras_from_pointmap(
        scene_pm,
        image_resolution=1,
        load_gt_images=True,
        max_img_size=1600,
        use_original_image_size=True,
        average_focal_distances=False,
        verbose=False,
    )
    print(f"[INFO] Created {len(cam_list)} cameras")
    pointmap_cameras = CamerasWrapper(cam_list, no_p3d_cameras=False)

    # Scale cameras
    target_scale = 5.0
    scale_factor = target_scale / pointmap_cameras.get_spatial_extent()
    pointmap_cameras = rescale_cameras(pointmap_cameras, scale_factor)

    # Step 3: Compute reference data
    print("[INFO] Step 3: Computing reference data from SfM points...")

    reference_data = torch.cat([
        pointmap_cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
            scale_factor * sfm_data['sfm_xyz'][sfm_data['image_sfm_points'][pointmap_cameras.gs_cameras[i_chart].image_name.split('.')[0]]]
        )[..., 2].view(scene_pm.points3d[i_chart][..., 0].shape)[None]
        for i_chart in range(len(pointmap_cameras))
    ], dim=0)

    print(f'[INFO] Reference data range: {reference_data.min():.3f} - {reference_data.max():.3f}')

    # Create masks
    masks = None

    # Step 4: Align depth maps
    print("[INFO] Step 4: Aligning depth maps using ParallelAligner...")
    output = align_charts_in_parallel(
        scene_pm,
        reference_data,
        masks=masks,
        rendering_size=1600,
        target_scale=target_scale,
        #verbose=True,
        return_training_losses=True,
        reprojection_matches_file=None,
        save_charts_data=True,
        #charts_data_path=charts_output_dir,
        **align_config,
    )

    # Step 5: Process alignment results
    print("[INFO] Step 5: Processing alignment results...")

    if align_config['use_learnable_confidence']:
        output_verts, output_depths, output_confs, training_losses = output
        output_confs = output_confs - 1.0
        print(f"[INFO] Depth range: {output_depths.min():.3f} - {output_depths.max():.3f}")
        print(f"[INFO] Confidence range: {output_confs.min():.3f} - {output_confs.max():.3f}")
    else:
        output_verts, output_depths, training_losses = output

    print(f"[INFO] Alignment complete!")
    print(f"[INFO] Output vertices shape: {output_verts.shape}")
    print(f"[INFO] Output depths shape: {output_depths.shape}")
    if align_config['use_learnable_confidence']:
        print(f"[INFO] Output confidence shape: {output_confs.shape}")

    # Save aligned depths and confidences
    if depth_output_dir is not None or conf_output_dir is not None:
        print("[INFO] Saving aligned depths and confidences...")

        if depth_output_dir is not None:
            depths_subdir = Path(depth_output_dir)
            ensure_dir(depths_subdir)

        if conf_output_dir is not None and align_config['use_learnable_confidence']:
            confs_subdir = Path(conf_output_dir)
            ensure_dir(confs_subdir)

        for i in range(len(pointmap_cameras)):
            image_name = pointmap_cameras.gs_cameras[i].image_name
            frame_match = image_name.split('_')

            try:
                if len(frame_match) >= 3:
                    frame_id_str = frame_match[2].split('.')[0]
                    frame_id = int(frame_id_str)
                elif len(frame_match) >= 2:
                    frame_id_str = frame_match[1].split('.')[0]
                    frame_id = int(frame_id_str)
                else:
                    frame_id = i
            except:
                frame_id = i

            cam_id = i
            frame_id = frame_id - 1

            depth_file_name = f"cam{cam_id:02d}_cam_{cam_id:04d}_{frame_id:04d}_depth.npy"
            conf_file_name = f"cam{cam_id:02d}_cam_{cam_id:04d}_{frame_id:04d}_conf.npy"

            # Save depth
            if depth_output_dir is not None:
                if isinstance(output_depths, torch.Tensor):
                    depth = output_depths[i].detach().cpu().numpy()
                else:
                    depth = output_depths[i].detach().cpu().numpy() if hasattr(output_depths[i], 'detach') else np.array(output_depths[i])

                if depth.ndim == 3 and depth.shape[0] == 1:
                    depth = depth[0]

                depth_path = depths_subdir / depth_file_name
                np.save(depth_path, depth.astype(np.float32))

            # Save confidence
            if conf_output_dir is not None and align_config['use_learnable_confidence']:
                if isinstance(output_confs, torch.Tensor):
                    conf = output_confs[i].detach().cpu().numpy()
                else:
                    conf = output_confs[i].detach().cpu().numpy() if hasattr(output_confs[i], 'detach') else np.array(output_confs[i])

                if conf.ndim == 3 and conf.shape[0] == 1:
                    conf = conf[0]

                conf_path = confs_subdir / conf_file_name
                np.save(conf_path, conf.astype(np.float32))

        print(f"[INFO] Successfully saved depths and confidences for {len(pointmap_cameras)} cameras")

    # Step 6: Create final aligned pointmap
    print("[INFO] Step 6: Creating final aligned pointmap...")

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

    return aligned_pointmap


def generate_camera_colors(n_cameras, color_palette='distinct', device='cuda'):
    """Generate distinct colors for each camera."""
    if color_palette == 'distinct':
        distinct_colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Violet
            [0.0, 0.5, 1.0],  # Sky blue
            [0.5, 1.0, 0.0],  # Yellow-green
            [1.0, 0.0, 0.5],  # Rose
            [0.0, 1.0, 0.5],  # Spring green
        ]

        colors = []
        for i in range(n_cameras):
            color_idx = i % len(distinct_colors)
            colors.append(distinct_colors[color_idx])

    elif color_palette == 'rainbow':
        colors = []
        for i in range(n_cameras):
            hue = i / max(1, n_cameras - 1)
            c = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append(list(c))

    elif color_palette == 'heatmap':
        colors = []
        for i in range(n_cameras):
            t = i / max(1, n_cameras - 1)
            r = t
            g = 0.0
            b = 1.0 - t
            colors.append([r, g, b])

    else:
        torch.manual_seed(42)
        colors = torch.rand(n_cameras, 3).tolist()

    return torch.tensor(colors, device=device, dtype=torch.float32)


def save_colored_pointcloud(pointmap: PointMapDepthAnything, output_dir: Path, color_palette: str = "distinct"):
    """Save colored point cloud with different colors for different cameras."""
    ensure_dir(output_dir)

    n_cameras = len(pointmap.points3d)
    colors = generate_camera_colors(n_cameras, color_palette, pointmap.points3d[0].device)

    all_points = []
    all_colors = []

    for cam_idx, img_path in enumerate(pointmap.img_paths):
        pts = pointmap.points3d[cam_idx]
        h, w = pts.shape[:2]
        pts_flat = pts.reshape(-1, 3)
        cam_color = colors[cam_idx].view(1, 3).expand(pts_flat.shape[0], -1)

        all_points.append(pts_flat)
        all_colors.append(cam_color)

    all_points = torch.cat(all_points, dim=0)
    all_colors = torch.cat(all_colors, dim=0)

    output_path = output_dir / "colored_pointcloud.npz"
    np.savez(
        output_path,
        points=all_points.cpu().numpy(),
        colors=all_colors.cpu().numpy(),
        camera_colors=colors.cpu().numpy(),
        n_cameras=n_cameras,
        color_palette=color_palette
    )

    print(f"[INFO] Colored point cloud saved to: {output_path}")
    print(f"[INFO] Total points: {all_points.shape[0]}")
    print(f"[INFO] Number of cameras: {n_cameras}")
    print(f"[INFO] Color palette: {color_palette}")

    for i in range(n_cameras):
        color = colors[i]
        print(f"  Camera {i}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(all_colors.cpu().numpy())

        ply_path = output_dir / "colored_pointcloud.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"[INFO] PLY format point cloud saved to: {ply_path}")

    except ImportError:
        print("[INFO] open3d not installed, cannot save PLY format")
    except Exception as e:
        print(f"[WARNING] Error saving PLY file: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-frame multi-camera SfM followed by DepthAnything V3 alignment"
    )
    parser.add_argument("dataset_root", type=Path, help="Root folder containing camera subdirectories")
    parser.add_argument("output_root", type=Path, help="Directory to store intermediate and output data")
    parser.add_argument(
        "--depthanything-model",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Depth Anything V3 model name from HuggingFace",
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

    # Alignment parameters
    parser.add_argument("--align-depths", action="store_true", help="Use ParallelAligner to align depth maps")
    parser.add_argument("--alignment-iterations", type=int, default=600, help="Number of alignment iterations")
    parser.add_argument("--use-confidence", action="store_true", help="Use learnable confidence during alignment")
    return parser.parse_args()


def main():
    args = parse_args()

    frame_dirs, all_frames = collect_multicam_frames(args.dataset_root)
    num_cameras = len(all_frames[0])
    print(f"[INFO] Number of cameras: {num_cameras}")

    start = max(args.start_frame, 0)
    stop = len(frame_dirs) if args.stop_frame is None else min(args.stop_frame, len(frame_dirs))
    if start >= stop:
        raise ValueError("Invalid frame range")

    ensure_dir(args.output_root)

    device = torch.device(args.device)

    # Load Depth Anything V3 model
    print(f"[INFO] Loading Depth Anything V3 model: {args.depthanything_model}")
    depthanything_model = load_model(
        model_name=args.depthanything_model,
        device=device,
    )

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
            print(f"[INFO] Processing camera {cam_idx}: {src}")
            dst = images_dir / f"cam{cam_idx:02d}_{src.name}"
            symlink_or_copy(src, dst)

        not_first_frame = False if frame_idx == 0 else True
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
            print(f"[INFO] Using ParallelAligner for depth alignment with Depth Anything V3...")
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

            pointmap = build_aligned_pointmap_with_depthanything_v3(
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
            print("[WARNING] Non-aligned mode not implemented for Depth Anything V3")
            print("[INFO] Please use --align-depths flag")
            continue

        if args.save_pointcloud:
            export_pointmap_to_pcd(pointmap, save_path=str(frame_root / "aligned_pointmap.ply"))

        if args.visualize:
            print(f"[INFO] Generating colored point cloud visualization...")
            colored_output_dir = frame_root / "colored_visualization"
            save_colored_pointcloud(pointmap, colored_output_dir, args.color_palette)

        if not args.keep_working_images:
            shutil.rmtree(images_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
