import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import yaml
import numpy as np
import shutil
from pathlib import Path

from matcha.pointmap.depthanythingv2 import get_pointmap_from_mast3r_scene_with_depthanything, export_pointmap_to_pcd
from matcha.dm_scene.cameras import CamerasWrapper, rescale_cameras, create_gs_cameras_from_pointmap
from matcha.dm_trainers.charts_alignment_temporal import align_charts_temporal

from rich.console import Console


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal charts alignment across multiple timestamps')

    # Scene arguments
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                       help='Base directory containing frame_XXXXX subdirectories (e.g., /path/to/try_use_first_frames_1119)')
    parser.add_argument('-o', '--output_path', type=str, default=None,
                       help='Output directory for aligned charts (default: {data_dir}/temporal_charts)')
    parser.add_argument('--mast3r_subdir', type=str, default='mast3r_sfm',
                       help='Subdirectory name containing MASt3R data (default: mast3r_sfm)')
    parser.add_argument('--depth_model', type=str, default="depthanythingv2")
    parser.add_argument('--white_background', type=bool, default=False)

    # Frame selection
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Starting frame index (default: 0)')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='Ending frame index (default: auto-detect all frames)')
    parser.add_argument('--frame_step', type=int, default=1,
                       help='Frame step size (default: 1, process every frame)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process (default: no limit)')

    # DepthAnything arguments
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')

    # Temporal parameters
    parser.add_argument('--temporal_encoding_type', type=str, default='learned', choices=['learned', 'positional'],
                       help='Type of temporal encoding (default: learned)')
    parser.add_argument('--temporal_encoding_dim', type=int, default=8,
                       help='Dimension of temporal features (default: 8)')

    # Deprecated arguments (kept for compatibility)
    parser.add_argument('--image_indices', type=str, default=None)
    parser.add_argument('--n_charts', type=int, default=None)

    # Config
    parser.add_argument('-c', '--config', type=str, default='temporal_default',
                       help='Configuration file name (default: temporal_default)')

    args = parser.parse_args()

    # Set console
    CONSOLE = Console(width=120)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Find all frame directories
    frame_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('frame_')])

    if len(frame_dirs) == 0:
        raise ValueError(f"No frame_* directories found in {data_dir}")

    CONSOLE.print(f"[INFO] Found {len(frame_dirs)} frame directories in {data_dir}")

    # Parse frame indices from directory names
    frame_indices = []
    for frame_dir in frame_dirs:
        try:
            frame_idx = int(frame_dir.name.split('_')[1])
            frame_indices.append((frame_idx, frame_dir))
        except (IndexError, ValueError):
            CONSOLE.print(f"[WARNING] Skipping invalid frame directory: {frame_dir.name}")
            continue

    # Sort by frame index
    frame_indices.sort(key=lambda x: x[0])

    # Filter frames based on arguments
    if args.end_frame is not None:
        frame_indices = [(idx, d) for idx, d in frame_indices if args.start_frame <= idx <= args.end_frame]
    else:
        frame_indices = [(idx, d) for idx, d in frame_indices if idx >= args.start_frame]

    # Apply frame step
    if args.frame_step > 1:
        frame_indices = frame_indices[::args.frame_step]

    # Apply max frames limit
    if args.max_frames is not None:
        frame_indices = frame_indices[:args.max_frames]

    n_timestamps = len(frame_indices)
    if n_timestamps == 0:
        raise ValueError("No frames to process after filtering")

    CONSOLE.print(f"[INFO] Processing {n_timestamps} timestamps:")
    for idx, frame_dir in frame_indices:
        CONSOLE.print(f"  - Frame {idx:05d}: {frame_dir.name}")

    # Set output path
    if args.output_path is None:
        args.output_path = data_dir
    else:
        args.output_path = Path(args.output_path)

    args.output_path.mkdir(parents=True, exist_ok=True)
    CONSOLE.print(f"\n[INFO] Output will be saved to: {args.output_path}")

    # Load config
    config_path = os.path.join('configs/charts_alignment', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pm_config = config['pointmap']
    scene_config = config['scene']
    align_config = config['alignment']
    masking_config = config['masking']

    # Reprojection loss
    if align_config['use_reprojection_loss']:
        raise NotImplementedError("Reprojection loss is not implemented yet for temporal alignment.")

    # === Build pointmaps for all timestamps ===
    CONSOLE.print("\n[INFO] Building pointmaps from MASt3R scenes...")

    # Load DepthAnything model once for reuse across all timestamps
    CONSOLE.print("[INFO] Loading DepthAnything model...")
    from matcha.pointmap.depthanythingv2 import load_model
    depth_model = load_model(
        checkpoint_dir=args.depthanythingv2_checkpoint_dir,
        encoder=args.depthanything_encoder,
        device=device,
    )
    CONSOLE.print("[INFO] Model loaded successfully. Will be reused for all timestamps.")

    temporal_scene_pms = []
    temporal_sfm_datas = []
    temporal_mast3r_pms = []

    for t, (frame_idx, frame_dir) in enumerate(frame_indices):
        CONSOLE.print(f"\n[Timestamp {t}/{n_timestamps-1}] Processing frame {frame_idx:05d}")

        # Construct paths
        mast3r_scene_path = frame_dir / args.mast3r_subdir
        source_path = mast3r_scene_path / 'images'  # Use images directory as source

        if not mast3r_scene_path.exists():
            raise ValueError(f"MASt3R scene not found: {mast3r_scene_path}")

        if not source_path.exists():
            raise ValueError(f"Images directory not found: {source_path}")

        # Build pointmap for this timestamp (reusing the depth model)
        scene_pm_t, sfm_data_t, mast3r_pm_t = get_pointmap_from_mast3r_scene_with_depthanything(
            scene_source_path=str(source_path),
            n_images_in_pointmap=args.n_charts,
            image_indices=args.image_indices,
            white_background=args.white_background,
            # MASt3R
            mast3r_scene_source_path=str(mast3r_scene_path),
            # DepthAnything
            depthanything_checkpoint_dir=args.depthanythingv2_checkpoint_dir,
            depthanything_encoder=args.depthanything_encoder,
            depth_model=depth_model,  # Pass the pre-loaded model
            # Misc
            device=device,
            return_sfm_data=True,
            return_mast3r_pointmap=True,
            **pm_config,
        )
        
        # Move all data to CPU immediately and clear GPU memory
        sfm_data_t['sfm_xyz'] = sfm_data_t['sfm_xyz'].cpu()
        sfm_data_t['sfm_col'] = sfm_data_t['sfm_col'].cpu()

        # Move pointmaps to CPU using move_everything_to_device (properly updates device attribute)
        scene_pm_t.move_everything_to_device('cpu')
        mast3r_pm_t.move_everything_to_device('cpu')
        scene_pm_t_cpu = scene_pm_t
        mast3r_pm_t_cpu = mast3r_pm_t
        CONSOLE.print(f"  Moved pointmaps to CPU. scene_pm device: {scene_pm_t_cpu.device}")

        # Save the pointmap for debugging/analysis
        pointmap_save_path = frame_dir / "pointmap_with_depth.ply"
        export_pointmap_to_pcd(scene_pm_t_cpu, save_path=str(pointmap_save_path))
        CONSOLE.print(f"  Saved pointmap to: {pointmap_save_path}")

        # Clean up camera objects that contain GPU tensors
        # These camera objects hold full-resolution images that can accumulate memory
        if hasattr(scene_pm_t_cpu, 'scene_cameras'):
            del scene_pm_t_cpu.scene_cameras
        if hasattr(scene_pm_t_cpu, 'scene_eval_cameras'):
            del scene_pm_t_cpu.scene_eval_cameras
        if hasattr(mast3r_pm_t_cpu, 'scene_cameras'):
            del mast3r_pm_t_cpu.scene_cameras
        if hasattr(mast3r_pm_t_cpu, 'scene_eval_cameras'):
            del mast3r_pm_t_cpu.scene_eval_cameras

        # Append CPU versions
        temporal_scene_pms.append(scene_pm_t_cpu)
        temporal_sfm_datas.append(sfm_data_t)
        temporal_mast3r_pms.append(mast3r_pm_t_cpu)

        # Clear GPU cache after each timestamp to prevent accumulation
        torch.cuda.empty_cache()

        # Print memory usage for debugging
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            CONSOLE.print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Clean up the depth model after all pointmaps are processed
    CONSOLE.print("\n[INFO] All pointmaps loaded. Cleaning up depth model...")
    del depth_model
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        CONSOLE.print(f"[INFO] Final GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # === Compute rescaling factor (using first timestamp as reference) ===
    _cam_list = create_gs_cameras_from_pointmap(
        temporal_scene_pms[0],
        image_resolution=1,
        load_gt_images=True,  # Keep True to avoid attribute errors
        max_img_size=pm_config['max_img_size'],
        use_original_image_size=True,
        average_focal_distances=False,
        verbose=False,
    )
    _pointmap_cameras = CamerasWrapper(_cam_list, no_p3d_cameras=False)
    _scale_factor = scene_config['target_scale'] / _pointmap_cameras.get_spatial_extent()

    # Clean up temporary camera objects
    del _cam_list, _pointmap_cameras
    torch.cuda.empty_cache()

    # === Prepare reference data for all timestamps ===
    CONSOLE.print("\n[INFO] Preparing reference data from SFM...")

    temporal_reference_data = []
    temporal_mast3r_masks = []

    for t in range(n_timestamps):
        scene_pm_t = temporal_scene_pms[t]
        sfm_data_t = temporal_sfm_datas[t]
        mast3r_pm_t = temporal_mast3r_pms[t]

        # Rescale cameras for this timestamp
        _cam_list_t = create_gs_cameras_from_pointmap(
            scene_pm_t,
            image_resolution=1,
            load_gt_images=True,  # Keep True to avoid attribute errors
            max_img_size=pm_config['max_img_size'],
            use_original_image_size=True,
            average_focal_distances=False,
            verbose=False,
        )
        _pointmap_cameras_t = CamerasWrapper(_cam_list_t, no_p3d_cameras=False)
        _pointmap_cameras_t = rescale_cameras(_pointmap_cameras_t, _scale_factor)

        # Move SFM data to GPU once for this timestamp
        sfm_xyz_gpu = sfm_data_t['sfm_xyz'].cuda()

        # Prepare reference data
        reference_data_t = torch.cat([
            _pointmap_cameras_t.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
                _scale_factor * sfm_xyz_gpu[sfm_data_t['image_sfm_points'][_pointmap_cameras_t.gs_cameras[i_chart].image_name.split('.')[0]]]
            )[..., 2].view(scene_pm_t.points3d[i_chart][..., 0].shape)[None]
            for i_chart in range(len(_pointmap_cameras_t))
        ], dim=0)

        # Move reference data to CPU to save GPU memory
        temporal_reference_data.append(reference_data_t.cpu())

        # Clean up temporary objects immediately
        del _cam_list_t, _pointmap_cameras_t, sfm_xyz_gpu, reference_data_t

        # Prepare masks if needed
        if masking_config['use_masks_for_alignment']:
            mast3r_mask_t = mast3r_pm_t_cpu.confidence > masking_config['sfm_mask_threshold']
            temporal_mast3r_masks.append(mast3r_mask_t)
            CONSOLE.print(f"[Timestamp {t}] {mast3r_mask_t.sum()} points in mask.")
        else:
            temporal_mast3r_masks.append(None)

        # Clear cache periodically
        if t % 4 == 0:
            torch.cuda.empty_cache()

    # Final cleanup before alignment
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        CONSOLE.print(f"[INFO] After preparing reference data - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    if masking_config['use_masks_for_alignment']:
        CONSOLE.print("[INFO] Using masks for alignment.")
    else:
        temporal_mast3r_masks = None
        CONSOLE.print("[INFO] All MASt3R-SfM points will be used for charts alignment.")

    # === Align the charts temporally ===
    CONSOLE.print("\n[INFO] Starting temporal charts alignment...")

    # Create temporary directory for charts data
    temp_charts_dir = args.output_path / "temp_charts"
    temp_charts_dir.mkdir(parents=True, exist_ok=True)

    output = align_charts_temporal(
        # Scene
        temporal_scene_pms=temporal_scene_pms,
        # Data parameters
        temporal_reference_data=temporal_reference_data,
        temporal_masks=temporal_mast3r_masks,
        rendering_size=pm_config['max_img_size'],
        target_scale=scene_config['target_scale'],
        # Temporal parameters
        temporal_encoding_type=args.temporal_encoding_type,
        temporal_encoding_dim=args.temporal_encoding_dim,
        # Other parameters
        verbose=True,
        return_training_losses=True,
        reprojection_matches_file=None,
        save_charts_data=True,
        charts_data_path=str(args.output_path),
        start_frame=args.start_frame,
        **align_config,
    )

    temporal_outputs, training_losses = output

    CONSOLE.print("\n===== Temporal Alignment Complete! =====")
    CONSOLE.print(f"\nProcessed frames:")

    # === Save depths and confidences in the same format as process_multicams_depth.py ===
    # for t, (frame_idx, frame_dir) in enumerate(frame_indices):
    #     output_verts_t, output_depths_t, output_confs_t = temporal_outputs[t]

    #     # Create output directories for this timestamp
    #     timestamp_dir = args.output_path / f"frame_{frame_idx:05d}"
    #     timestamp_dir.mkdir(parents=True, exist_ok=True)

    #     charts_dir = timestamp_dir / "charts"
    #     depth_dir = timestamp_dir / "depth"
    #     conf_dir = timestamp_dir / "confidence"

    #     charts_dir.mkdir(parents=True, exist_ok=True)
    #     depth_dir.mkdir(parents=True, exist_ok=True)
    #     conf_dir.mkdir(parents=True, exist_ok=True)

    #     CONSOLE.print(f"\n[Frame {frame_idx:05d} -> Timestamp {t}]")
    #     CONSOLE.print(f"  Output vertices shape: {output_verts_t.shape}")
    #     CONSOLE.print(f"  Output depths shape: {output_depths_t.shape}")
    #     if align_config['use_learnable_confidence']:
    #         CONSOLE.print(f"  Output confidence shape: {output_confs_t.shape}")

    #     # Move charts data file to the correct directory
    #     # The align_charts_temporal function may save files like timestamp_{t:04d}_charts_data.npz
    #     charts_data_file = temp_charts_dir / f"timestamp_{t:04d}_charts_data.npz"
    #     if charts_data_file.exists():
    #         dest_charts_file = charts_dir / "charts_data.npz"
    #         shutil.move(str(charts_data_file), str(dest_charts_file))
    #         CONSOLE.print(f"  Moved charts data to: {dest_charts_file}")

    #     # Get cameras for this timestamp
    #     scene_pm_t = temporal_scene_pms[t]
    #     _cam_list_t = create_gs_cameras_from_pointmap(
    #         scene_pm_t,
    #         image_resolution=1,
    #         load_gt_images=True,
    #         max_img_size=pm_config['max_img_size'],
    #         use_original_image_size=True,
    #         average_focal_distances=False,
    #         verbose=False,
    #     )
    #     _pointmap_cameras_t = CamerasWrapper(_cam_list_t, no_p3d_cameras=False)

    #     # Save depths and confidences for each camera
    #     CONSOLE.print(f"  Saving {len(_pointmap_cameras_t)} camera outputs...")

    #     for i_chart in range(len(_pointmap_cameras_t)):
    #         # Get image name and parse camera/frame info
    #         image_name = _pointmap_cameras_t.gs_cameras[i_chart].image_name

    #         # Parse camera ID from image name (e.g., "cam00_..." -> cam_id=0)
    #         try:
    #             if image_name.startswith('cam'):
    #                 cam_id_str = image_name.split('_')[0].replace('cam', '')
    #                 cam_id = int(cam_id_str)
    #             else:
    #                 cam_id = i_chart
    #         except:
    #             cam_id = i_chart

    #         # Construct file names matching process_multicams_depth.py format
    #         # Format: cam{cam_id:02d}_cam_{cam_id:04d}_{frame_id:04d}_depth.npy
    #         depth_file_name = f"cam{cam_id:02d}_cam_{cam_id:04d}_{frame_idx:04d}_depth.npy"
    #         conf_file_name = f"cam{cam_id:02d}_cam_{cam_id:04d}_{frame_idx:04d}_conf.npy"

    #         # Extract and save depth
    #         if isinstance(output_depths_t, torch.Tensor):
    #             depth = output_depths_t[i_chart].detach().cpu().numpy()
    #         else:
    #             depth = output_depths_t[i_chart].detach().cpu().numpy() if hasattr(output_depths_t[i_chart], 'detach') else np.array(output_depths_t[i_chart])

    #         # Ensure 2D array (H, W)
    #         if depth.ndim == 3 and depth.shape[0] == 1:
    #             depth = depth[0]

    #         depth_path = depth_dir / depth_file_name
    #         np.save(depth_path, depth.astype(np.float32))

    #         # Extract and save confidence
    #         if align_config['use_learnable_confidence']:
    #             if isinstance(output_confs_t, torch.Tensor):
    #                 conf = output_confs_t[i_chart].detach().cpu().numpy()
    #             else:
    #                 conf = output_confs_t[i_chart].detach().cpu().numpy() if hasattr(output_confs_t[i_chart], 'detach') else np.array(output_confs_t[i_chart])

    #             # Ensure 2D array (H, W)
    #             if conf.ndim == 3 and conf.shape[0] == 1:
    #                 conf = conf[0]

    #             conf_path = conf_dir / conf_file_name
    #             np.save(conf_path, conf.astype(np.float32))

        # CONSOLE.print(f"  Saved to:")
        # CONSOLE.print(f"    Charts: {charts_dir}")
        # CONSOLE.print(f"    Depths: {depth_dir}")
        # CONSOLE.print(f"    Confidence: {conf_dir}")

    # Clean up temporary charts directory
    if temp_charts_dir.exists():
        shutil.rmtree(temp_charts_dir, ignore_errors=True)

    CONSOLE.print(f"\n[INFO] All results saved to: {args.output_path}")
