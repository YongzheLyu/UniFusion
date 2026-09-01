import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from matcha.dm_scene.parallel_aligner_temporal import (
    ParallelAlignerTemporal,
    TemporalEncodingParams,
)
from matcha.dm_scene.parallel_aligner import (
    MLPParams,
    ChartsEncodingParams,
    DepthEncodingParams,
    MultiResChartsEncodingParams,
)
from matcha.dm_scene.meshes import get_manifold_meshes_from_pointmaps
from matcha.dm_scene.cameras import CamerasWrapper, create_gs_cameras_from_pointmap, rescale_cameras
from matcha.pointmap.mast3r import load_mast3r_matches


def align_charts_temporal(
    # Scene - temporal version takes a list of scene_pms
    temporal_scene_pms,  # List of PointMaps, length T
    # Data parameters
    temporal_reference_data,  # List of reference data tensors, length T
    temporal_masks=None,  # List of masks, length T (optional)
    rendering_size=1600,
    target_scale=5.,
    # Temporal parameters
    temporal_encoding_type='positional',  # 'learned' or 'positional'
    temporal_encoding_dim=8,
    # ParallelAligner architecture parameters
    use_learnable_depth_encoding=True,
    learnable_depth_encoding_mode='add',
    predict_in_disparity_space=False,
    use_learnable_confidence=True,
    use_meta_mlp=False,
    use_lora_mlp=False,
    lora_rank=4,
    n_lora_layers=2,
    resfield_rank=8,
    use_multi_res_charts_encoding=True,
    # ParallelAligner optimization parameters
    n_iterations=1000,
    use_gradient_loss=False,
    use_hessian_loss=False,
    use_normal_loss=True,
    use_curvature_loss=True,
    use_matching_loss=True,
    use_reprojection_loss=False,
    use_occlusion_loss=False,
    use_ssi_loss=False,
    matching_thr_factor=1./20.,
    matching_update_iters=None,
    use_confidence_in_matching_loss=False,
    weight_encodings_with_confidence=False,
    regularize_chart_encodings_norms=False,
    use_total_variation_on_depth_encodings=False,
    gradient_loss_weight=25.,
    hessian_loss_weight=50.,
    normal_loss_weight=4.,
    curvature_loss_weight=1.,
    reprojection_loss_weight=2.,
    occlusion_loss_weight=5.0,
    depth_order_loss_type="hinge",
    ssi_loss_weight=2.0,
    reprojection_loss_power=0.5,
    reprojection_matches_file=None,
    matching_loss_weight=5.,
    chart_encodings_norm_loss_weight=2.,
    total_variation_on_depth_encodings_weight=5.0,
    encodings_lr=1e-2,
    mlp_lr=1e-3,
    confidence_lr=1e-3,
    lr_update_iters=[1000],
    lr_update_factor=0.1,
    verbose=True,
    return_training_losses=False,
    save_charts_data=True,
    charts_data_path='./',
    start_frame=0,
):
    """Align multiple charts across multiple timestamps.

    This function extends align_charts_in_parallel to handle temporal sequences.
    Key idea: Treat T timestamps with N charts each as T×N total charts, where
    charts from the same view across different timestamps share MLP heads.

    Args:
        temporal_scene_pms: List of PointMaps, one per timestamp (length T)
        temporal_reference_data: List of reference depth tensors (length T)
        temporal_masks: List of masks (length T), optional
        rendering_size (int, optional): Maximum image size. Defaults to 1600.
        target_scale (float, optional): Target scene scale. Defaults to 5.
        temporal_encoding_type (str, optional): 'learned' or 'positional'. Defaults to 'learned'.
        temporal_encoding_dim (int, optional): Dimension of temporal features. Defaults to 8.
        ... (other parameters same as align_charts_in_parallel)

    Returns:
        Tuple containing aligned outputs for all timestamps
    """
    print("so weird!!!!!!!!!!")
    n_timestamps = len(temporal_scene_pms)
    device = temporal_scene_pms[0].points3d.device
    print("device:", device)
    if verbose:
        print(f"\n===== Temporal Charts Alignment =====")
        print(f"Number of timestamps: {n_timestamps}")

    # Get basic info from first timestamp
    first_pm = temporal_scene_pms[0]
    n_charts_per_timestamp = first_pm.points3d.shape[0]
    pm_h, pm_w = first_pm.points3d.shape[1:3]

    if verbose:
        print(f"Charts per timestamp: {n_charts_per_timestamp}")
        print(f"Total charts (T×N): {n_timestamps * n_charts_per_timestamp}")
        print(f"Pointmap resolution: {pm_h} × {pm_w}\n")

    # Initialize parameter objects
    charts_encoding_params = ChartsEncodingParams()
    depth_encoding_params = DepthEncodingParams()
    mlp_params = MLPParams()
    temporal_encoding_params = TemporalEncodingParams(
        encoding_dim=temporal_encoding_dim,
        encoding_type=temporal_encoding_type,
    )

    if verbose:
        print("===== ParallelAlignerTemporal parameters =====\n")
        print("Charts encoding dim", charts_encoding_params.encoding_dim)
        print("Charts encoding resolution factor", charts_encoding_params.resolution_factor)
        print("Charts encoding initialization range", charts_encoding_params.initialization_range, '\n')
        print("Depth encoding dim", depth_encoding_params.encoding_dim)
        print("Depth encoding n bins", depth_encoding_params.n_bins)
        print("Depth encoding initialization range", depth_encoding_params.initialization_range, '\n')
        print("Temporal encoding dim", temporal_encoding_params.encoding_dim)
        print("Temporal encoding type", temporal_encoding_params.encoding_type, '\n')
        print("MLP n deformation layers", mlp_params.n_deformation_layers)
        print("MLP deformation layer size", mlp_params.deformation_layer_size, '\n')
        print("ResField rank", resfield_rank, '\n')

    # === Build combined data across all timestamps ===
    # Memory optimization: keep data on CPU until needed
    all_cameras = []
    all_initial_depths = []
    all_reference_depths = []
    all_masks = []
    cam_list = []
    
    for t in range(n_timestamps):
        scene_pm_t = temporal_scene_pms[t]

        # Build cameras for this timestamp
        cam_list_t = create_gs_cameras_from_pointmap(
            scene_pm_t,
            image_resolution=1,
            load_gt_images=True,
            max_img_size=rendering_size,
            use_original_image_size=True,
            average_focal_distances=False,
            verbose=False,
        )
        pointmap_cameras_t = CamerasWrapper(cam_list_t, no_p3d_cameras=False)
        cam_list.extend(cam_list_t)
        
        # Compute scale factor (use first timestamp as reference)
        if t == 0:
            if target_scale is not None:
                scale_factor = target_scale / pointmap_cameras_t.get_spatial_extent()
            else:
                scale_factor = 1.
            spatial_extent = pointmap_cameras_t.get_spatial_extent()
        
        # Rescale cameras
        pointmap_cameras_t = rescale_cameras(pointmap_cameras_t, scale_factor)
        
        # Move cameras to CPU if they're on GPU to save memory
        # They will be moved back to GPU when needed in ParallelAligner
        if hasattr(pointmap_cameras_t, 'p3d_cameras'):
            # Keep cameras on CPU for now
            pass

        # Build low-res cameras
        # lowres_cameras_t = CamerasWrapper.from_p3d_cameras(
        #     p3d_cameras=pointmap_cameras_t.p3d_cameras,
        #     height=pm_h,
        #     width=pm_w,
        # )
        
        # Build initial depths (memory-optimized: keep on CPU until needed)
        # Avoid creating large GPU tensors during initialization
        # Move points3d to CPU if it's on GPU to reduce memory pressure
        if hasattr(scene_pm_t.points3d, 'device') and scene_pm_t.points3d.device.type == 'cuda':
            pt_maps_t = (scale_factor * scene_pm_t.points3d).cpu()
        else:
            pt_maps_t = scale_factor * scene_pm_t.points3d
        
        imgs_t = scene_pm_t.images
        
        # Create mesh only to extract vertices, then immediately delete
        # Process on CPU to avoid GPU memory spikes
        manifolds_t, _ = get_manifold_meshes_from_pointmaps(
            pt_maps_t, imgs_t, masks=None, return_single_mesh_object=True, return_manifold_idx=True
        )
        _verts_t = manifolds_t.verts_packed().clone().cpu()  # Keep on CPU
        # Immediately delete mesh to free memory
        del manifolds_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Compute initial depths on CPU to save GPU memory
        initial_depths_t = torch.cat([
            pointmap_cameras_t.p3d_cameras[i_chart].get_world_to_view_transform().cpu().transform_points(
                _verts_t.reshape(scene_pm_t.points3d.shape)[i_chart].reshape(-1, 3)
            )[..., 2].reshape(1, pm_h, pm_w) for i_chart in range(len(pointmap_cameras_t))
        ], dim=0)  # (N, H, W) - on CPU
        
        # Delete temporary vertices and clear cache
        del _verts_t, pt_maps_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Save initial depth maps as heatmaps for this timestamp
        # if True:
        #     initial_depth_save_dir = "initial_depth_heatmaps"
        #     os.makedirs(initial_depth_save_dir, exist_ok=True)

        #     # initial_depths_t shape: (N, H, W)
        #     n_charts, height, width = initial_depths_t.shape

        #     # Save each chart's depth map
        #     for chart_idx in range(n_charts):
        #         depth_map = initial_depths_t[chart_idx].cpu().detach().numpy()

                # Create heatmap
                # plt.figure(figsize=(10, 6))
                # plt.imshow(depth_map, cmap='plasma', aspect='auto')
                # plt.colorbar()
                # plt.title(f'Timestamp {t}, Chart {chart_idx} - Initial Depth Map')
                # plt.axis('off')

                # # Save the heatmap
                # save_path = os.path.join(initial_depth_save_dir, f'timestamp_{t:03d}_chart_{chart_idx:02d}_initial_depth.png')
                # plt.savefig(save_path, bbox_inches='tight', dpi=150)
                # plt.close()

                # print(f"Saved initial depth heatmap: {save_path}")

        #print(len(lowres_cameras_t), len(pointmap_cameras_t))
        #all_cameras.append(lowres_cameras_t)
        all_initial_depths.append(initial_depths_t)
        all_reference_depths.append(temporal_reference_data[t])

        if temporal_masks is not None:
                # Ensure masks are on CPU to save GPU memory
            mask_t = temporal_masks[t]
            if hasattr(mask_t, 'device') and mask_t.device.type == 'cuda':
                all_masks.append(mask_t.cpu())
            else:
                all_masks.append(mask_t)
        
        # Cleanup: delete temporary image data if it's a copy
        # (scene_pm_t.images is usually needed, so be careful)
        if hasattr(imgs_t, 'device') and imgs_t.device.type == 'cuda' and imgs_t is not scene_pm_t.images:
            del imgs_t
        
        # Periodic memory cleanup every few timestamps
        if (t + 1) % max(1, n_timestamps // 10) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if verbose:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"[MEMORY] After timestamp {t+1}/{n_timestamps}: {allocated:.2f} GB allocated")
    print(len(cam_list))
    pointmap_cameras = CamerasWrapper(cam_list, no_p3d_cameras=False)
    pointmap_cameras = rescale_cameras(pointmap_cameras, scale_factor)
    
    # Create lowres cameras - keep cameras on CPU initially if possible
    # The ParallelAligner will move them to GPU when needed
    lowres_cameras = CamerasWrapper.from_p3d_cameras(
        p3d_cameras=pointmap_cameras.p3d_cameras,
        height=pm_h,
        width=pm_w,
    )
    
    # Clear intermediate camera data if not needed
    del cam_list
    torch.cuda.empty_cache()
    # Combine all timestamps - keep on CPU initially to save GPU memory
    # These will be moved to GPU inside ParallelAlignerTemporal if needed
    combined_initial_depths = torch.cat(all_initial_depths, dim=0)  # (T×N, H, W) - on CPU
    combined_reference_depths = torch.cat(all_reference_depths, dim=0)  # (T×N, H, W) - on CPU

    if temporal_masks is not None:
        combined_masks = torch.cat(all_masks, dim=0)  # (T×N, H, W) - on CPU
    else:
        combined_masks = None
    
    # Clear intermediate lists to free memory (but keep the combined tensors)
    # Note: We keep the combined tensors but they're on CPU
    del all_initial_depths, all_reference_depths, all_masks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose:
        print(f"[MEMORY] Combined depths shape: {combined_initial_depths.shape}")
        print(f"[MEMORY] Combined depths device: {combined_initial_depths.device}")
        print(f"[MEMORY] Combined reference depths device: {combined_reference_depths.device}")

    # Combine cameras into a single CamerasWrapper
    # all_p3d_cameras = []
    # all_gs_cameras = []
    # for cameras_t in all_cameras:
    #     all_p3d_cameras.extend(cameras_t.p3d_cameras)
    #     all_gs_cameras.extend(cameras_t.gs_cameras)

    # combined_cameras = CamerasWrapper(all_gs_cameras, no_p3d_cameras=False)
    # combined_cameras.p3d_cameras = all_p3d_cameras
    #combined_cameras = all_cameras

    # Compute matching threshold
    matching_thr = matching_thr_factor * spatial_extent

    # Load matches if needed (TODO: extend to temporal case)
    if use_reprojection_loss and reprojection_matches_file is not None:
        match_to_img, match_to_pix, idx_to_image = load_mast3r_matches(reprojection_matches_file)
    else:
        match_to_img, match_to_pix, idx_to_image = None, None, None
    
    # Final memory cleanup before creating ParallelAlignerTemporal
    # This is critical to maximize available GPU memory for optimization
    torch.cuda.empty_cache()
    if verbose:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"[MEMORY] Before ParallelAlignerTemporal init:")
            print(f"  > Allocated: {allocated:.2f} GB")
            print(f"  > Reserved: {reserved:.2f} GB")
    
    # === Create ParallelAlignerTemporal ===
    # Note: ParallelAlignerTemporal will move depths to GPU during initialization
    # We keep them on CPU here to save memory until needed
    pa = ParallelAlignerTemporal(
        depths=combined_initial_depths,
        cameras=lowres_cameras,
        n_timestamps=n_timestamps,
        n_charts_per_timestamp=n_charts_per_timestamp,
        temporal_encoding_params=temporal_encoding_params,
        charts_encoding_params=ChartsEncodingParams(),
        depth_encoding_params=DepthEncodingParams(),
        mlp_params=MLPParams(),
        use_learnable_depth_encoding=use_learnable_depth_encoding,
        learnable_depth_encoding_mode=learnable_depth_encoding_mode,
        use_learnable_confidence=use_learnable_confidence,
        device=device,
        predict_in_disparity_space=predict_in_disparity_space,
        use_meta_mlp=use_meta_mlp,
        use_lora_mlp=use_lora_mlp,
        lora_rank=lora_rank,
        n_lora_layers=n_lora_layers,
        resfield_rank=resfield_rank,
        use_multi_res_charts_encoding=use_multi_res_charts_encoding,
        multi_res_charts_encoding_params=MultiResChartsEncodingParams(),
        weight_encodings_with_confidence=weight_encodings_with_confidence,
        pm_h=pm_h,
        pm_w=pm_w,
    )
    #pause = input("pause")
    # === Optimize ===
    pa.optimize(
        reference_data=combined_reference_depths,
        masks=combined_masks,
        n_iterations=n_iterations,
        use_gradient_loss=use_gradient_loss,
        use_hessian_loss=use_hessian_loss,
        use_normal_loss=use_normal_loss,
        use_matching_loss=use_matching_loss,
        use_curvature_loss=use_curvature_loss,
        use_reprojection_loss=use_reprojection_loss,
        use_occlusion_loss=use_occlusion_loss,
        use_ssi_loss=use_ssi_loss,
        regularize_chart_encodings_norms=regularize_chart_encodings_norms,
        use_total_variation_on_depth_encodings=use_total_variation_on_depth_encodings,
        matching_thr=matching_thr,
        use_confidence_in_matching_loss=use_confidence_in_matching_loss,
        matching_update_iters=matching_update_iters,
        gradient_loss_weight=gradient_loss_weight,
        hessian_loss_weight=hessian_loss_weight,
        normal_loss_weight=normal_loss_weight,
        curvature_loss_weight=curvature_loss_weight,
        matching_loss_weight=matching_loss_weight,
        reprojection_loss_weight=reprojection_loss_weight,
        occlusion_loss_weight=occlusion_loss_weight,
        depth_order_loss_type=depth_order_loss_type,
        ssi_loss_weight=ssi_loss_weight,
        reprojection_loss_power=reprojection_loss_power,
        chart_encodings_norm_loss_weight=chart_encodings_norm_loss_weight,
        total_variation_on_depth_encodings_weight=total_variation_on_depth_encodings_weight,
        encodings_lr=encodings_lr,
        mlp_lr=mlp_lr,
        confidence_lr=confidence_lr,
        lr_update_iters=lr_update_iters,
        lr_update_factor=lr_update_factor,
        verbose=verbose,
        match_to_img=match_to_img,
        match_to_pix=match_to_pix,
    )

    # === Extract outputs for each timestamp ===
    output_verts = pa._deformed_verts.clone().cuda()  # (T×N, H, W, 3)
    print(output_verts.device)
    output_depths = torch.cat([
        pa.cameras.p3d_cameras[i_chart].get_world_to_view_transform().cuda().transform_points(
            output_verts[i_chart].reshape(-1, 3)
        )[..., 2].reshape(1, pm_h, pm_w) for i_chart in range(len(pa.cameras))
    ], dim=0)  # (T×N, H, W)

    if use_learnable_confidence:
        with torch.no_grad():
            output_confs = pa.confidence  # (T×N, H, W)
    else:
        output_confs = 4. * torch.ones_like(output_depths)

    # Reorganize outputs by timestamp
    temporal_outputs = []
    for t in range(n_timestamps):
        start_idx = t * n_charts_per_timestamp
        end_idx = (t + 1) * n_charts_per_timestamp

        output_verts_t = output_verts[start_idx:end_idx]  # (N, H, W, 3)
        output_depths_t = output_depths[start_idx:end_idx]  # (N, H, W)
        output_confs_t = output_confs[start_idx:end_idx]  # (N, H, W)
        
        temporal_outputs.append((output_verts_t, output_depths_t, output_confs_t))

    # Save charts data for each timestamp
    if save_charts_data:
        for t in range(n_timestamps):
            output_verts_t, output_depths_t, output_confs_t = temporal_outputs[t]

            # 为每个timestamp创建子目录
            timestamp_dir = os.path.join(charts_data_path, f"frame_{t+start_frame:05d}")
            depth_dir = os.path.join(timestamp_dir, "depth")
            conf_dir = os.path.join(timestamp_dir, "confidence")
            charts_dir = os.path.join(timestamp_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            if use_learnable_confidence:
                os.makedirs(conf_dir, exist_ok=True)

            print(f"[INFO] Saving charts data for timestamp {t} to {timestamp_dir}")

            # 获取当前timestamp的cameras信息
            scene_pm_t = temporal_scene_pms[t]
            n_charts = len(scene_pm_t.images)

            # 遍历每个相机chart，按照process_multicams_depth的格式保存
            for i_chart in range(n_charts):
                # 从相机图像名称解析帧ID和相机ID
                image_name = scene_pm_t.img_paths[i_chart]
                # 假设图像名称格式为: camXX_frame_XXXXX.jpg 或类似格式
                basename = os.path.basename(image_name)
                frame_match = basename.split('_')

                try:
                    if len(frame_match) >= 3:
                        # 格式: camXX_frame_XXXXX.jpg -> 提取XXXXX
                        frame_id_str = frame_match[2].split('.')[0]
                        frame_id = int(frame_id_str) - 1  # 转换为0-based索引
                    elif len(frame_match) >= 2:
                        # 备用格式
                        frame_id_str = frame_match[1].split('.')[0]
                        frame_id = int(frame_id_str) - 1
                    else:
                        frame_id = i_chart
                except:
                    frame_id = i_chart

                # 相机ID就是chart的索引
                cam_id = i_chart

                # 构建文件名，格式: cam{cam_id:02d}_cam_{cam_id:04d}_{frame_id:04d}_depth.npy
                depth_file_name = f"cam{cam_id:02d}_cam_{cam_id:04d}_{frame_id+start_frame:04d}_depth.npy"
                conf_file_name = f"cam{cam_id:02d}_cam_{cam_id:04d}_{frame_id+start_frame:04d}_conf.npy"

                # 保存深度
                depth = output_depths_t[i_chart].detach().cpu().numpy()
                # 确保是2D数组 (H, W)
                if depth.ndim == 3 and depth.shape[0] == 1:
                    depth = depth[0]

                depth_path = os.path.join(depth_dir, depth_file_name)
                np.save(depth_path, depth.astype(np.float32))

                # 保存置信度
                if use_learnable_confidence:
                    conf = output_confs_t[i_chart].detach().cpu().numpy()
                    # 确保是2D数组 (H, W)
                    if conf.ndim == 3 and conf.shape[0] == 1:
                        conf = conf[0]

                    conf_path = os.path.join(conf_dir, conf_file_name)
                    np.save(conf_path, conf.astype(np.float32))

            # Save charts_data.npz for this timestamp
            
            charts_data_path_t = os.path.join(charts_dir, "charts_data.npz")
            print(f"[INFO] Saving charts data for timestamp {t} to {charts_data_path_t}")

            # Get data for this timestamp
            # Extract prior depths from combined tensor (which is on CPU)
            start_idx = t * n_charts_per_timestamp
            end_idx = (t + 1) * n_charts_per_timestamp
            prior_depths_t = combined_initial_depths[start_idx:end_idx]  # (N, H, W)
            output_depths_t_data = output_depths_t  # (N, H, W)
            output_verts_t_data = output_verts_t  # (N, H, W, 3)
            output_confs_t_data = output_confs_t  # (N, H, W)

            # Save timestamp-specific charts data
            np.savez(
                charts_data_path_t,
                prior_depths=prior_depths_t.cpu().numpy() if prior_depths_t.device.type == 'cuda' else prior_depths_t.numpy(),
                depths=output_depths_t_data.cpu().numpy(),
                pts=output_verts_t_data.cpu().numpy(),
                confs=output_confs_t_data.cpu().numpy(),
                scale_factor=scale_factor,
                timestamp=t,
                n_charts_per_timestamp=n_charts_per_timestamp,
            )

            print(f"[INFO] Successfully saved depths, confidences, and charts data for {n_charts} cameras at timestamp {t}")

    

    if use_learnable_confidence:
        if return_training_losses:
            return temporal_outputs, pa.train_losses
        else:
            return temporal_outputs
    else:
        if return_training_losses:
            return temporal_outputs, pa.train_losses
        else:
            return temporal_outputs
