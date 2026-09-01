import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import yaml
import numpy as np
import pickle
from pathlib import Path

from matcha.pointmap.depthanythingv2 import get_pointmap_from_mast3r_scene_with_depthanything
from matcha.dm_scene.cameras import CamerasWrapper, rescale_cameras, create_gs_cameras_from_pointmap

from rich.console import Console


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess temporal data for charts alignment across multiple timestamps')

    # Scene arguments
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                       help='Base directory containing frame_XXXXX subdirectories (e.g., /path/to/try_use_first_frames_1119)')
    parser.add_argument('-o', '--output_path', type=str, default=None,
                       help='Output path for preprocessed data (default: {data_dir}/preprocessed_temporal_data.pkl)')

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
    # DepthAnything V2
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    # DepthAnything V3 (DA3)，仅当 --depth_model depthanythingv3 时使用
    parser.add_argument(
        '--depthanythingv3_model',
        type=str,
        default='depth-anything/DA3NESTED-GIANT-LARGE',
        help='Depth Anything V3 (DA3) model name when using depthanythingv3 backend',
    )

    # Scene arguments
    parser.add_argument('--mast3r_subdir', type=str, default='mast3r_sfm',
                       help='Subdirectory name containing MASt3R data (default: mast3r_sfm)')
    parser.add_argument('--depth_model', type=str, default="depthanythingv2")
    parser.add_argument('--white_background', type=bool, default=False)

    # Deprecated arguments (kept for compatibility)
    parser.add_argument('--image_indices', type=str, default=None)
    parser.add_argument('--n_charts', type=int, default=4)

    # Config
    parser.add_argument('-c', '--config', type=str, default='temporal_default',
                       help='Configuration file name (default: temporal_default)')

    # Preprocessing options
    parser.add_argument('--use_masks', action='store_true', default=False,
                       help='Whether to prepare masks for alignment')

    # Memory optimization options
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Number of frames to process per batch to save memory (default: process all at once)')

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
        args.output_path = data_dir / "preprocessed_temporal_data.pkl"
    else:
        args.output_path = Path(args.output_path)

    # Load config
    config_path = os.path.join('configs/charts_alignment', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pm_config = config['pointmap']
    scene_config = config['scene']
    masking_config = config['masking']

    CONSOLE.print("\n[INFO] Starting temporal data preprocessing...")

    # Determine if using batched processing for memory efficiency
    use_batched_processing = args.batch_size is not None and args.batch_size > 0
    if use_batched_processing:
        CONSOLE.print(f"[INFO] Using batched processing with batch size: {args.batch_size}")
        batch_size = min(args.batch_size, n_timestamps)  # Don't exceed total frames
    else:
        CONSOLE.print("[INFO] Processing all frames at once (no batching)")
        batch_size = n_timestamps

    # === Build pointmaps for all timestamps ===
    CONSOLE.print("\n[INFO] Building pointmaps from MASt3R scenes...")

    # 根据 depth_model 选择后端；保持整体 data flow 不变
    depth_model = None
    depth_model_type = args.depth_model.lower()
    if depth_model_type == "depthanythingv2":
        CONSOLE.print("[INFO] Using DepthAnything V2 backend...")
        # Load DepthAnything V2 model once for reuse across all timestamps
        CONSOLE.print("[INFO] Loading DepthAnything V2 model...")
        from matcha.pointmap.depthanythingv2 import load_model
        depth_model = load_model(
            checkpoint_dir=args.depthanythingv2_checkpoint_dir,
            encoder=args.depthanything_encoder,
            device=device,
        )
        CONSOLE.print("[INFO] DepthAnything V2 model loaded successfully. Will be reused for all timestamps.")
    elif depth_model_type == "depthanythingv3":
        CONSOLE.print("[INFO] Using DepthAnything V3 (DA3) backend...")
        # 复用同一个 DA3 模型，避免在每个 timestamp 内重复加载
        from matcha.pointmap.depthanythingv3 import load_model as load_model_da3
        depth_model = load_model_da3(
            model_name=args.depthanythingv3_model,
            device=device,
        )
        CONSOLE.print("[INFO] DepthAnything V3 model loaded successfully. Will be reused for all timestamps.")
    else:
        raise ValueError(f"Unknown depth_model: {args.depth_model}. Supported: depthanythingv2, depthanythingv3")

    # Data structures to collect final results
    all_temporal_scene_pms = []
    all_temporal_reference_data = []
    all_temporal_mast3r_masks = []
    all_temporal_frame_indices = []
    scale_factor = None

    # Process frames in batches
    for batch_start in range(0, n_timestamps, batch_size):
        batch_end = min(batch_start + batch_size, n_timestamps)
        batch_frames = frame_indices[batch_start:batch_end]
        batch_idx = batch_start // batch_size

        CONSOLE.print(f"\n{'='*60}")
        CONSOLE.print(f"[INFO] Processing batch {batch_idx+1} (frames {batch_start} to {batch_end-1})")
        CONSOLE.print(f"{'='*60}")

        # Data structures for this batch
        temporal_scene_pms = []
        temporal_sfm_datas = []
        temporal_mast3r_pms = []
        temporal_frame_indices = []

        for t, (frame_idx, frame_dir) in enumerate(batch_frames):
            CONSOLE.print(f"\n[Batch {batch_idx+1}, Timestamp {t}/{len(batch_frames)-1}] Processing frame {frame_idx:05d}")

            # Construct paths（目录结构保持不变）
            mast3r_scene_path = frame_dir / args.mast3r_subdir
            source_path = mast3r_scene_path / 'images'  # images 仍然放在 mast3r_sfm/images 下

            if not mast3r_scene_path.exists():
                raise ValueError(f"MASt3R scene not found: {mast3r_scene_path}")

            if not source_path.exists():
                raise ValueError(f"Images directory not found: {source_path}")

            # 根据 backend 选择不同的 DepthAnything 实现，但输出接口保持一致
            if depth_model_type == "depthanythingv2":
                # ----- 原有 DepthAnything V2 流程 -----
                scene_pm_t, sfm_data_t, mast3r_pm_t = get_pointmap_from_mast3r_scene_with_depthanything(
                    scene_source_path=str(source_path),
                    image_indices=args.image_indices,
                    white_background=args.white_background,
                    n_images_in_pointmap=args.n_charts,
                    # MASt3R
                    mast3r_scene_source_path=str(mast3r_scene_path),
                    # DepthAnything V2
                    depthanything_checkpoint_dir=args.depthanythingv2_checkpoint_dir,
                    depthanything_encoder=args.depthanything_encoder,
                    depth_model=depth_model,  # 复用已加载的模型
                    # Misc
                    device=device,
                    return_sfm_data=True,
                    return_mast3r_pointmap=True,
                    **pm_config,
                )
            else:
                # ----- 新增：DepthAnything V3 (DA3) 流程 -----
                # 不改变整体 data flow，只是换成 DA3 的 pointmap 构建函数。
                from matcha.pointmap.depthanythingv3 import get_pointmap_from_mast3r_scene_with_depthanything as get_pointmap_da3

                scene_pm_t, sfm_data_t, mast3r_pm_t = get_pointmap_da3(
                    # 与 v2 一样，使用 mast3r_sfm/images 作为 scene_source_path，
                    # 函数内部会自动判断是否需要附加 "images" 子目录。
                    scene_source_path=str(source_path),
                    n_images_in_pointmap=args.n_charts,
                    image_indices=args.image_indices,
                    white_background=args.white_background,
                    # MASt3R
                    mast3r_scene_source_path=str(mast3r_scene_path),
                    # DA3 模型（复用）
                    depthanything_model_name=args.depthanythingv3_model,
                    depth_model=depth_model,
                    # Misc
                    device=str(device),
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
            if depth_model_type == "depthanythingv2":
                from matcha.pointmap.depthanythingv2 import export_pointmap_to_pcd
            else:
                from matcha.pointmap.depthanythingv3 import export_pointmap_to_pcd
            #export_pointmap_to_pcd(scene_pm_t_cpu, save_path=str(pointmap_save_path))
            #CONSOLE.print(f"  Saved pointmap to: {pointmap_save_path}")

            # Clean up camera objects that contain GPU tensors
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
            temporal_frame_indices.append(frame_idx)

                # Clear GPU cache after each timestamp to prevent accumulation
            torch.cuda.empty_cache()

                # Print memory usage for debugging
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                CONSOLE.print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            # === Compute rescaling factor (using first frame of first batch) ===
            if scale_factor is None:
                CONSOLE.print("\n[INFO] Computing rescaling factor using first frame...")
                _cam_list = create_gs_cameras_from_pointmap(
                    temporal_scene_pms[t],
                    image_resolution=1,
                    load_gt_images=True,
                    max_img_size=pm_config['max_img_size'],
                    use_original_image_size=True,
                    average_focal_distances=False,
                    verbose=False,
                )
                _pointmap_cameras = CamerasWrapper(_cam_list, no_p3d_cameras=False)
                
                scale_factor = scene_config['target_scale'] / _pointmap_cameras.get_spatial_extent()

                del _cam_list, _pointmap_cameras
                torch.cuda.empty_cache()
                CONSOLE.print(f"[INFO] Scale factor: {scale_factor}")

            # === Prepare reference data for this batch ===
            CONSOLE.print(f"\n[INFO] Preparing reference data for batch {batch_idx+1}...")
            temporal_reference_data = []
            temporal_mast3r_masks = []

        for t in range(len(batch_frames)):
            scene_pm_t = temporal_scene_pms[t]
            sfm_data_t = temporal_sfm_datas[t]
            mast3r_pm_t = temporal_mast3r_pms[t]

            # Rescale cameras for this timestamp
            _cam_list_t = create_gs_cameras_from_pointmap(
                scene_pm_t,
                image_resolution=1,
                load_gt_images=True,
                max_img_size=pm_config['max_img_size'],
                use_original_image_size=True,
                average_focal_distances=False,
                verbose=False,
            )
            _pointmap_cameras_t = CamerasWrapper(_cam_list_t, no_p3d_cameras=False)
            _pointmap_cameras_t = rescale_cameras(_pointmap_cameras_t, scale_factor)

            # Move SFM data to GPU once for this timestamp
            sfm_xyz_gpu = sfm_data_t['sfm_xyz'].cuda()

            # Prepare reference data
            reference_data_t = torch.cat([
                _pointmap_cameras_t.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
                    scale_factor * sfm_xyz_gpu[sfm_data_t['image_sfm_points'][_pointmap_cameras_t.gs_cameras[i_chart].image_name.split('.')[0]]]
                )[..., 2].view(scene_pm_t.points3d[i_chart][..., 0].shape)[None]
                for i_chart in range(len(_pointmap_cameras_t))
            ], dim=0)

            # Move reference data to CPU to save GPU memory
            temporal_reference_data.append(reference_data_t.cpu())

            # Clean up temporary objects immediately
            del _cam_list_t, _pointmap_cameras_t, sfm_xyz_gpu, reference_data_t

            # Prepare masks if needed
            if args.use_masks or masking_config['use_masks_for_alignment']:
                mast3r_mask_t = mast3r_pm_t.confidence > masking_config['sfm_mask_threshold']
                temporal_mast3r_masks.append(mast3r_mask_t)
                CONSOLE.print(f"[Batch {batch_idx+1}, Frame {t}] {mast3r_mask_t.sum()} points in mask.")
            else:
                temporal_mast3r_masks.append(None)

        # Accumulate results from this batch
        all_temporal_scene_pms.extend(temporal_scene_pms)
        all_temporal_reference_data.extend(temporal_reference_data)
        all_temporal_mast3r_masks.extend(temporal_mast3r_masks)
        all_temporal_frame_indices.extend(temporal_frame_indices)

        # Clean up batch data (keep sfm_data and mast3r_pm for debugging, but clear from memory)
        del temporal_scene_pms, temporal_sfm_datas, temporal_mast3r_pms
        del temporal_reference_data, temporal_mast3r_masks, temporal_frame_indices
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            CONSOLE.print(f"[INFO] After batch {batch_idx+1} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Clean up the depth model after all pointmaps are processed
    CONSOLE.print("\n[INFO] All pointmaps loaded. Cleaning up depth model...")
    if depth_model is not None:
        del depth_model
        torch.cuda.empty_cache()

    # Update variables for final output
    temporal_scene_pms = all_temporal_scene_pms
    temporal_reference_data = all_temporal_reference_data
    temporal_mast3r_masks = all_temporal_mast3r_masks
    temporal_frame_indices = all_temporal_frame_indices

    # === Save all preprocessed data ===
    CONSOLE.print(f"\n[INFO] Saving preprocessed data to: {args.output_path}")

    preprocessed_data = {
        'temporal_scene_pms': temporal_scene_pms,
        #'temporal_sfm_datas': temporal_sfm_datas,
        #'temporal_mast3r_pms': temporal_mast3r_pms,
        'temporal_reference_data': temporal_reference_data,
        'temporal_mast3r_masks': temporal_mast3r_masks,
        'temporal_frame_indices': temporal_frame_indices,
        'scale_factor': scale_factor,
        'pm_config': pm_config,
        'scene_config': scene_config,
        #'masking_config': masking_config,
        'args': vars(args),
    }

    # Save using pickle
    with open(args.output_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)

    CONSOLE.print(f"[INFO] Preprocessed data saved successfully!")

    # Print summary
    CONSOLE.print("\n===== Preprocessing Complete! =====")
    CONSOLE.print(f"Processed {n_timestamps} timestamps")
    CONSOLE.print(f"Data saved to: {args.output_path}")
    CONSOLE.print(f"Scale factor: {scale_factor}")

    # Memory cleanup
    try:
        del temporal_scene_pms
        del temporal_sfm_datas
        del temporal_mast3r_pms
        del temporal_reference_data, temporal_mast3r_masks
   
        torch.cuda.empty_cache()
    except Exception as e:
        CONSOLE.print(f"[WARNING] Error during memory cleanup: {e}")
