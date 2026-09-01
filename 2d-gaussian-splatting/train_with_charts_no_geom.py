# 重构后的训练脚本：分成 coarse 与 fine 两阶段
# （基于你提供的脚本，做最小改动以实现阶段化）
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

import os
import sys
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import gc
import copy
import open3d as o3d
import torch
from random import randint
import random
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import SH2RGB
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.point_utils import depth_to_normal
from matcha.dm_scene.charts import (
    load_charts_data,
    build_priors_from_charts_data,
    schedule_regularization_factor_1,
    schedule_regularization_factor_2,
    get_gaussian_parameters_from_charts_data,
)
from matcha.dm_regularization.depth import compute_depth_order_loss
from matcha.dm_utils.rendering import normal2curv, depth2normal_parallel, normal2curv_parallel
import numpy as np
from pathlib import Path
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import matplotlib.pyplot as plt

def set_random_seed(seed: int):
    """
    设置所有随机种子以确保实验可复现

    Args:
        seed: 随机种子值
    """
    print(f"[INFO] Setting random seed to {seed} for reproducible results")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # PyTorch backend determinism (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed (for reproducible hashing)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Verify seed setting
    print(f"[DEBUG] Python random: {random.randint(0, 100)}")
    print(f"[DEBUG] NumPy random: {np.random.randint(0, 100)}")
    print(f"[DEBUG] PyTorch random: {torch.randint(0, 100, (1,)).item()}")

# New confidence -> weight
def confidence_to_weight(confidence:torch.Tensor):
    conf_weights = confidence - 1.
    return torch.sigmoid((conf_weights - 2.) * 2.)


def load_preprocessed_priors_from_dir(preprocessed_priors_dir, cameras, device="cpu", frames=10):
    """
    从预处理的priors目录逐个加载prior数据，现场生成normals和curvs而不是读取文件

    Args:
        preprocessed_priors_dir: 预处理priors的根目录
        cameras: 相机列表，用于确定加载哪些文件
        device: 目标设备

    Returns:
        dict: 包含所有prior数据的字典，格式与build_priors_from_charts_data相同
    """
    priors_dir = Path(preprocessed_priors_dir)

    # 初始化存储结构
    depths_list = []
    confs_list = []
    normals_list = []
    curvs_list = []
    prior_depths_list = []

    # 从metadata加载元数据
    metadata_path = priors_dir / "metadata.npy"
    if metadata_path.exists():
        metadata = np.load(metadata_path, allow_pickle=True).item()
        scale_factor = metadata.get('scale_factor', 1.0)
    else:
        prior_path = priors_dir / "charts_data.npz"
        if prior_path.exists():
            charts_data = np.load(prior_path)
            scale_factor = charts_data.get('scale_factor', 1.0)
        else:
            print(f"[WARNING] 未找到charts数据文件: {prior_path}")
            scale_factor = np.array(1.0)
        #scale_factor = 1.0
        print(f"[WARNING] 未找到metadata文件: {metadata_path}")
    scale_factor = torch.from_numpy(scale_factor).to(device)
    print(f"[INFO] 从 {preprocessed_priors_dir} 加载预处理priors并现场生成normals/curvs...")
    #print()
    # 为每个相机加载对应的prior文件
    for i, cam in enumerate(cameras):
        # 从相机名称解析相机ID和帧ID
        frame_match = cam.image_name.split('_')
        cam_id = None
        if len(frame_match) >= 3:
            try:
                # 例如 cam_0001_0101.jpg -> cam_id=1, frame_id=101
                cam_id = int(frame_match[1])
                frame_id = int(frame_match[2].split('.')[0])
            except Exception:
                frame_id = i + 1
        else:
            frame_id = i + 1

        # 如果无法从文件名中可靠解析 cam_id，则退化为按索引编号
        if cam_id is None:
            cam_id = i + 1

        print(cam.image_name, frame_id)
        # 只将 cam_id 调整为 0-based，frame_id 保持原始编号（不再强行从 0 开始）
        cam_id = cam_id - 1
        #11.4 debug cam_id -1
        # 构建文件名
        file_name = f"cam_{cam_id:04d}_{frame_id:04d}.npy"
        #print(file_name)
        try:
            # 加载各个prior文件
            depth_path = priors_dir / "depths" / file_name
            conf_path = priors_dir / "confs" / file_name

            if depth_path.exists():
                depth = np.load(depth_path)
                depths_list.append(torch.from_numpy(depth).to(device))
            else:
                print(f"[WARNING] 未找到深度文件: {depth_path}")
                depths_list.append(torch.zeros((1, cam.image_height, cam.image_width)).to(device))

            if conf_path.exists():
                conf = np.load(conf_path)
                confs_list.append(torch.from_numpy(conf).to(device))
            else:
                print(f"[WARNING] 未找到置信度文件: {conf_path}")
                confs_list.append(torch.ones((1, cam.image_height, cam.image_width)).to(device))

            # 现场生成法向量和曲率（而不是从文件读取）
            current_depth = depths_list[-1]
            current_conf = confs_list[-1]
            
            # 确保深度图有正确的形状 (1, H, W)
            if current_depth.dim() == 2:
                current_depth = current_depth.unsqueeze(0)
            if current_conf.dim() == 2:
                current_conf = current_conf.unsqueeze(0)
            
            # 现场生成法向量
            try:
                # 使用相机参数和深度图生成法向量
                world_view_transform = cam.world_view_transform.unsqueeze(0).to(device)
                full_proj_transform = cam.full_proj_transform.unsqueeze(0).to(device)
                
                # 使用并行函数生成法向量
                normals = depth2normal_parallel(
                    depths=current_depth.unsqueeze(0),  # (1, 1, H, W)
                    world_view_transforms=world_view_transform,
                    full_proj_transforms=full_proj_transform
                )  # (1, H, W, 3)
                
                # 转换形状从 (1, H, W, 3) 到 (3, H, W)
                normals = normals.squeeze(0).permute(2, 0, 1)  # (3, H, W)
                normals_list.append(normals)
                
                # 现场生成曲率
                # 创建掩码（基于置信度）
                mask = (current_conf > 0.5).float()  # (1, H, W)
                
                # 使用法向量生成曲率
                curvatures = normal2curv_parallel(
                    normal=normals.unsqueeze(0),  # (1, 3, H, W)
                    mask=mask.unsqueeze(0)        # (1, 1, H, W)
                )  # (1, 1, H, W)
                
                # 转换形状从 (1, 1, H, W) 到 (1, H, W)
                curvatures = curvatures.squeeze(0)  # (1, H, W)
                curvs_list.append(curvatures)
                
            except Exception as e:
                print(f"[WARNING] 现场生成normals/curvs失败 for camera {i}: {e}")
                # 使用零值作为后备
                normals_list.append(torch.zeros((3, cam.image_height, cam.image_width)).to(device))
                curvs_list.append(torch.zeros((1, cam.image_height, cam.image_width)).to(device))

            # prior_depths使用与depths相同的数据
            prior_depths_list.append(depths_list[-1])

        except Exception as e:
            print(f"[ERROR] 加载相机 {i} 的prior文件失败: {e}")
            # 使用默认值
            depths_list.append(torch.zeros((1, cam.image_height, cam.image_width)).to(device))
            confs_list.append(torch.ones((1, cam.image_height, cam.image_width)).to(device))
            normals_list.append(torch.zeros((3, cam.image_height, cam.image_width)).to(device))
            curvs_list.append(torch.zeros((1, cam.image_height, cam.image_width)).to(device))
            prior_depths_list.append(torch.zeros((1, cam.image_height, cam.image_width)).to(device))

    print(f"[INFO] 成功加载 {len(cameras)} 个相机的priors并现场生成normals/curvs")

    # 保存prior depth可视化
    # output_dir = os.path.join(os.path.dirname(preprocessed_priors_dir), "prior_depth_visualizations")
    # os.makedirs(output_dir, exist_ok=True)
    # print(f"[INFO] 保存prior depth可视化到: {output_dir}")

    # for i, (cam, prior_depth) in enumerate(zip(cameras, prior_depths_list)):
    #     # 从相机名称解析信息
    #     frame_match = cam.image_name.split('_')
    #     if len(frame_match) >= 2:
    #         try:
    #             frame_id = int(frame_match[2].split('.')[0])
    #             cam_id = int(frame_match[1])
    #         except:
    #             frame_id = i + 1
    #             cam_id = 0
    #     else:
    #         frame_id = i + 1
    #         cam_id = 0

    #     # 生成文件名
    #     save_name = f"cam_{cam_id:04d}_frame_{frame_id:04d}.png"
    #     save_path = os.path.join(output_dir, save_name)

    #     # 可视化并保存
    #     depth_np = prior_depth.squeeze().cpu().numpy()

    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(depth_np, cmap="Spectral")
    #     plt.colorbar(label="Depth")
    #     plt.title(f"Prior Depth - {cam.image_name}")
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     plt.close()

    #     if (i + 1) % 10 == 0:
    #         print(f"[INFO] 已保存 {i + 1}/{len(cameras)} 个prior depth可视化")

    # print(f"[INFO] 所有prior depth可视化已保存到: {output_dir}")

    # 构建与原始格式兼容的字典
    charts_priors = {
        'scale_factor': scale_factor,
        'prior_depths': prior_depths_list,
        'depths': depths_list,
        'confs': confs_list,
        'normals': normals_list,
        'curvs': curvs_list
    }

    return charts_priors

def scene_reconstruction(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    use_refined_charts,
    use_mip_filter,
    dense_data_path,
    use_chart_view_every_n_iter,
    normal_consistency_from,
    distortion_from,
    depthanythingv2_checkpoint_dir,
    depthanything_encoder,
    dense_regul,
    stage: str,
    train_iter: int,
    gaussians: GaussianModel,
    scene: Scene,
    tb_writer=None,
    frames=10,
    warmup_mode: bool = False,
    warmup_frame_sampling: str = "all"
):
    """
    单阶段训练函数，可供 coarse 或 fine 调用：
      - stage: "coarse" 或 "fine"
      - train_iter: 本阶段的最大迭代次数（相对全局iteration起点）
      - checkpoint: 如果提供，尝试恢复模型并设置 first_iter
    """
    save_log_images = True
    save_log_images_every_n_iter = 500

    # 注意：本函数假定 gaussians 已经用 charts 初始化（见 training()）
    # 但这里也要做 training_setup / checkpoint 恢复等


    # 如果有 dense supervision，dense_viewpoint_cams 已在 training() 中准备并放在 dataset 属性中
    use_dense_supervision = getattr(dataset, "use_dense_supervision", False)
    dense_viewpoint_cams = getattr(dataset, "dense_viewpoint_cams", None)

    # 在coarse阶段，只使用第一帧的相机
    if stage == "coarse":
        use_first_frame_only = True
        print("[INFO] Coarse stage: Using only first frame cameras closed!!!!")
    else:
        use_first_frame_only = False
        print("[INFO] Fine stage: Using all frames")

    # 在warmup阶段，根据warmup_frame_sampling参数采样帧
    if warmup_mode:
        print(f"[INFO] Warmup stage: Frame sampling mode = {warmup_frame_sampling}")

    gaussians.training_setup(opt)
    first_iter = 0
    if checkpoint:
        # 从 checkpoint 恢复（如果 checkpoint 有 stage 信息也可在外部做判断）
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_idx_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_prior_depth_for_log = 0.0
    ema_prior_normal_for_log = 0.0
    ema_prior_curvature_for_log = 0.0
    ema_prior_anisotropy_for_log = 0.0

    # Get training cameras
    viewpoint_cams = scene.getTrainCameras()

    # 在coarse阶段，只使用第一帧的相机（每个相机的第一个时间步）
    if use_first_frame_only:
        # 根据frames参数，每个相机有frames张图片
        # 选择每个相机的第一帧（索引0, frames, 2*frames, ...）
        first_frame_cams = []
        for i in range(0, len(viewpoint_cams)):
            print(viewpoint_cams[i].time, viewpoint_cams[i].image_name)
            if viewpoint_cams[i].time == 0:
                first_frame_cams.append(viewpoint_cams[i])
        viewpoint_cams = first_frame_cams
        print(f"[INFO] Coarse stage: Filtered to {len(viewpoint_cams)} cameras (first frame only)")

    # 在warmup阶段，根据warmup_frame_sampling参数采样帧
    if warmup_mode:
        if warmup_frame_sampling == "first_10_percent":
            # 使用前10%帧
            num_cams = len(viewpoint_cams)
            num_selected = max(1, int(num_cams * 0.1))
            viewpoint_cams = viewpoint_cams[:num_selected]
            print(f"[INFO] Warmup stage: Using first 10% ({num_selected}/{num_cams}) cameras")
        elif warmup_frame_sampling == "uniform_every_10":
            # 每十帧均匀采样一帧
            viewpoint_cams = viewpoint_cams[::10]
            print(f"[INFO] Warmup stage: Using every 10th frame ({len(viewpoint_cams)} cameras)")
        elif warmup_frame_sampling == "all":
            # 使用全部帧
            print(f"[INFO] Warmup stage: Using all {len(viewpoint_cams)} cameras")
        else:
            raise ValueError(f"Unknown warmup_frame_sampling: {warmup_frame_sampling}")

    #print()
    # Set mip filter if requested
    if use_mip_filter:
        print("[INFO] Using mip filter during training.")
        gaussians.set_mip_filter(use_mip_filter)
        gaussians.compute_mip_filter(cameras=dense_viewpoint_cams if use_dense_supervision else viewpoint_cams)

    # Build priors from charts data (assume already prepared globally; but safe to call)
    # ---- these were prepared in training() earlier, we fetch from dataset
    # Add logging about camera split and ensure proper variable scoping
    print("=" * 60)
    print(f"[CAMERA SPLIT SUMMARY]")
    print(f"  Training cameras: {len(scene.getTrainCameras())}")
    print(f"  Testing cameras: {len(scene.getTestCameras())}")

    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()

    # if len(train_cameras) > 0:
    #     print("  Training camera names:")
    #     for i, cam in enumerate(train_cameras):
    #         print(f"    {i+1}: {cam.image_name} ")
    # if len(test_cameras) > 0:
    #     print("  Testing camera names:")
    #     for i, cam in enumerate(test_cameras):
    #         print(f"    {i+1}: {cam.image_name} ")
    print("=" * 60)

    if len(train_cameras) == 0:
        print("[ERROR] No training cameras loaded! Please check your data structure.")
        return

    preprocessed_priors_dir = dataset.source_path + "/preprocessed_priors"
    if preprocessed_priors_dir:
        print(f"[INFO] 从预处理目录加载priors: {preprocessed_priors_dir}")
        # Use training cameras only for building priors
        train_cameras = viewpoint_cams


        if len(train_cameras) > 0:
            print(f"[INFO] Building charts priors from {len(train_cameras)} training cameras only")
            # Also pass the filtered charts_data to ensure consistency
            
            charts_priors = load_preprocessed_priors_from_dir(
                preprocessed_priors_dir,
                train_cameras,  # Use all training cameras
                device="cuda",
                frames=frames if not use_first_frame_only else 1
            )
        else:
            # Fallback to all cameras if no train cameras found
            print("[WARNING] No training cameras found, falling back to all cameras")
            charts_priors = load_preprocessed_priors_from_dir(
                preprocessed_priors_dir,
                viewpoint_cams,
                device="cuda",
                frames=frames
            )
        #dataset.setattr("charts_priors", charts_priors)
    #charts_priors = getattr(dataset, "charts_priors", None)
    if charts_priors is None:
        print("[ERROR] No charts priors")
        # fallback: build from charts_data in dataset.source_path
        # charts_data_path = f'{dataset.source_path}/charts_data.npz'
        # charts_data = load_charts_data(charts_data_path)
        # charts_priors = build_priors_from_charts_data(charts_data, viewpoint_cams)

    # 使用新的逐个加载方式
    
    charts_scale_factor = charts_priors['scale_factor']
    charts_prior_depths = charts_priors['prior_depths']
    charts_depths = charts_priors['depths']
    print(len(charts_depths))
    charts_confs = charts_priors['confs']
    charts_normals = charts_priors['normals']
    charts_curvs = charts_priors['curvs']
    if False:
        for idx, depth in enumerate(charts_depths):
            os.makedirs('./test_depth',exist_ok=True)
            plt.figure(figsize=(3, 4))
            
            
            plt.title("Charts Depth")
            plt.imshow(depth.squeeze().cpu().numpy(), cmap="Spectral")
            plt.colorbar()
            plt.savefig(f'./test_depth/{idx}.jpg')
            plt.close()
    #time.sleep(10000)
    # prepare progress bar: note we run from first_iter..train_iter
    sample_render_pkg = render(viewpoint_cams[0], gaussians, pipe, background, stage='coarse')
    target_depth_shape = sample_render_pkg['surf_depth'].shape[-2:]  # H, W
    print(f"[INFO] surf depth range: min {sample_render_pkg['surf_depth'].min()} max {sample_render_pkg['surf_depth'].max()}")
    
    target_normal_shape = sample_render_pkg['rend_normal'].shape[1:]  # C, H, W
    
    
    def upsample_tensor(tensor_list, target_size, unsqueeze_channel=False):
        """
        使用 CPU 和 numpy 进行上采样/下采样
        - tensor_list: list of torch.Tensor (H,W) 或 (C,H,W)
        - target_size: (H_out, W_out)
        - unsqueeze_channel: True 时把 tensor 当作单通道
        """
        resized_list = []
        H_out, W_out = target_size

        for t in tensor_list:
            t_cpu = t.detach().cpu().numpy()
            if unsqueeze_channel and t_cpu.ndim == 2:
                t_cpu = t_cpu[None, :, :]  # (1,H,W)
            C = t_cpu.shape[0] if t_cpu.ndim == 3 else 1

            # 用 numpy 的简单 resize (bilinear)
            resized = np.zeros((C, H_out, W_out), dtype=t_cpu.dtype)
            for c in range(C):
                resized[c] = np.array(
                    Image.fromarray(t_cpu[c]).resize((W_out, H_out), resample=Image.BILINEAR)
                )
            resized_list.append(torch.from_numpy(resized).cpu())
        return resized_list
    if True:
        charts_depths_up = upsample_tensor(charts_depths, target_depth_shape, True)
        charts_normals_up = upsample_tensor(charts_normals, target_normal_shape, False)
        charts_confs_up = upsample_tensor(charts_confs, target_depth_shape, True)
        charts_curvs_up = upsample_tensor(charts_curvs, target_normal_shape, False)
        charts_prior_depths_up = upsample_tensor(charts_prior_depths, target_depth_shape, True)
    else:
        charts_depths_up = charts_depths
        charts_normals_up = charts_normals
        charts_confs_up = charts_confs
        charts_curvs_up = charts_curvs
        charts_prior_depths_up = charts_prior_depths
    print('='*50)
    print(charts_scale_factor)
    print('='*50)
    charts_depths_up = [depth / charts_scale_factor.cpu() for depth in charts_depths_up] 
    charts_prior_depths_up = [depth / charts_scale_factor.cpu() for depth in charts_prior_depths_up] 
    charts_priors['depths'] = charts_depths_up
    charts_priors['normals'] = charts_normals_up
    charts_priors['confs'] = charts_confs_up
    charts_priors['curvs'] = charts_curvs_up
    print(f"[INFO] charts depth range: min {charts_depths_up[0].min()} max {charts_depths_up[0].max()}")
    torch.cuda.empty_cache()
    progress_bar = tqdm(range(first_iter, train_iter), desc=f"Training ({stage})")
    first_iter += 1

    # main loop: 注意此处 final_iter = train_iter
    final_iter = train_iter
    warmup_iter = 0
    if stage == "fine" and opt.freeze_gaussians_in_fine:
        opt.densify_until_iter += warmup_iter
    #加一个1000的迭代，用于冻结/解冻高斯参数
    for iteration in range(first_iter, final_iter + warmup_iter + 1 if stage == "fine" and opt.freeze_gaussians_in_fine else final_iter + 1):
        iter_start.record()

        # Warm-up mode: Freeze Gaussian parameters (only train deformation MLP)
        if warmup_mode:
            if iteration == first_iter:
                print("[INFO] Warm-up mode: Freezing Gaussian parameters, only training deformation MLP")
                # Freeze all Gaussian parameters
                for param_group in gaussians.optimizer.param_groups:
                    if param_group["name"] in ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]:
                        param_group["lr"] = 0.0
                # Keep deformation network parameters trainable
                for param_group in gaussians.optimizer.param_groups:
                    if param_group["name"] not in ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]:
                        # Reset learning rate for deformation network parameters
                        param_group["lr"] = param_group.get("initial_lr", param_group["lr"])
        # if stage == "fine":
        #     for param_group in gaussians.optimizer.param_groups:
        #         if param_group["name"] in [ "f_dc", "f_rest", "opacity"]:
        #             param_group["lr"] = 0.0
        # Freeze/unfreeze Gaussian parameters in fine stage (legacy code, kept for reference)
        # if stage == "fine" and opt.freeze_gaussians_in_fine:

        # increase SH degree occasionally (跟你原来一致)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # pick camera: charts vs dense sampling
        if iteration % use_chart_view_every_n_iter == 0:
            if not viewpoint_idx_stack:
                viewpoint_idx_stack = list(range(len(viewpoint_cams)))
            viewpoint_idx = viewpoint_idx_stack.pop(randint(0, len(viewpoint_idx_stack)-1))
            viewpoint_cam = viewpoint_cams[viewpoint_idx]
        else:
            # dense branch
            if not use_dense_supervision:
                # safety fallback
                viewpoint_idx = randint(0, len(viewpoint_cams)-1)
                viewpoint_cam = viewpoint_cams[viewpoint_idx]
            else:
                dense_viewpoint_idx_stack = getattr(dataset, "dense_viewpoint_idx_stack", None)
                if not dense_viewpoint_idx_stack:
                    dense_viewpoint_idx_stack = list(range(len(dense_viewpoint_cams)))
                    dataset.dense_viewpoint_idx_stack = dense_viewpoint_idx_stack
                viewpoint_idx = dataset.dense_viewpoint_idx_stack.pop(randint(0, len(dataset.dense_viewpoint_idx_stack)-1))
                viewpoint_cam = dense_viewpoint_cams[viewpoint_idx]
        
        gaussians.update_learning_rate(iteration)
      
        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage = 'coarse' if stage == "coarse" else 'fine')
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        
        prior_depth = charts_depths_up[viewpoint_idx].cuda()  # CPU tensor (C,H,W) 或 (H,W)
        prior_conf  = charts_confs_up[viewpoint_idx].cuda()
        prior_norm  = charts_normals_up[viewpoint_idx].cuda()
        prior_curv  = charts_curvs_up[viewpoint_idx].cuda()

        # regularization schedule (一样) - ALL GEOMETRY SUPERVISION DISABLED
        lambda_normal = 0.0  # DISABLED
        lambda_dist = 0.0  # DISABLED

        rend_dist = render_pkg.get("rend_dist", None)
        rend_normal = render_pkg.get('rend_normal', None)
        surf_normal = render_pkg.get('surf_normal', None)
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None] if rend_normal is not None else 0.0
        normal_loss = lambda_normal * (normal_error).mean() if rend_normal is not None else 0.0
        dist_loss = lambda_dist * (rend_dist).mean() if rend_dist is not None else 0.0

        total_loss = loss + dist_loss + normal_loss

        # chart / dense regularization (与原逻辑一致) - ALL GEOMETRY SUPERVISION DISABLED
        surf_depth = render_pkg.get('surf_depth', None)
        total_regularization_loss = 0.
        lambda_anisotropy = 1.0  # DISABLED
        anisotropy_max_ratio = 5.
        #print("use_chart:", use_chart_view_every_n_iter)
        #use chart view = 1
        if iteration % use_chart_view_every_n_iter == 0:
            rend_curvature = normal2curv(render_pkg['rend_normal'], torch.ones_like(render_pkg['rend_normal'][0:1]))

            initial_regularization_factor = 0.5
            regularization_factor = schedule_regularization_factor_2(iteration, initial_regularization_factor)
            #regularization_factor = 0.0
            #debug 12.15
            lambda_prior_depth = 0.0  # DISABLED
            lambda_prior_depth_derivative = 0.0  # DISABLED
            lambda_prior_normal = 0.0  # DISABLED
            lambda_prior_curvature = 0.0  # DISABLED
            # if iteration < 3000 and stage == "fine":
            #     lambda_prior_depth = 0.0
            #     lambda_prior_depth_derivative = 0.0
            #     lambda_prior_normal = 0.0
            #     lambda_prior_curvature = 0.0
            confidence_weighting = 0.5

            depths_to_regularize = [surf_depth, render_pkg.get('rend_depth', None)]
        
            
            depth_prior_loss = lambda_prior_depth * (
                confidence_weighting * prior_conf * torch.log(1. + charts_scale_factor * (prior_depth - surf_depth).abs())
             ).mean()
            #print(surf_normal.shape, charts_normals_up[viewpoint_idx].shape)
            
            if lambda_prior_depth_derivative > 0:
                depth_prior_loss += (
                    lambda_prior_depth_derivative
                    * torch.exp(-(prior_conf - 1.) ** 2 / 2.)
                    * (1. - (surf_normal * prior_norm).sum(dim=0))
                ).mean()
                # depth_prior_loss += (
                #     lambda_prior_depth_derivative
                #     * torch.exp(-(prior_conf - 1.) ** 2 / 2.)
                #     * (1. - (surf_normal * prior_norm.permute(2,0,1)).sum(dim=0))
                # ).mean()
            #print(depth_prior_loss.item())
            normal_prior_loss = lambda_prior_normal * (1. - (rend_normal * prior_norm).sum(dim=0)).mean()
            #normal_prior_loss = lambda_prior_normal * (1. - (rend_normal * prior_norm.permute(2,0,1)).sum(dim=0)).mean()
            curv_prior_loss = lambda_prior_curvature * (prior_curv - rend_curvature).abs().mean()
            #normal_prior_loss = torch.zeros_like(loss.detach())
            # depth order regularization (scheduling kept)
            depth_order_prior_loss = torch.zeros_like(loss.detach())
            
            if True:
                lambda_depth_order = 0.0  # DISABLED - depth order regularization
                if not (stage == "fine" and opt.freeze_gaussians_in_fine):
                    # All lambda_depth_order scheduling DISABLED
                    # if iteration > 1_500:
                    #     lambda_depth_order = 1.
                    # if iteration > 3_000:
                    #     lambda_depth_order = 0.1
                    # if iteration > 4_500:
                    #     lambda_depth_order = 0.01
                    # if iteration > 6_000:
                    #     lambda_depth_order = 0.001
                    pass
                else:
                    # All lambda_depth_order scheduling DISABLED
                    # if iteration > warmup_iter + 1500:
                    #     lambda_depth_order = 1.
                    # if iteration > warmup_iter + 3_000:
                    #     lambda_depth_order = 0.1
                    # if iteration > warmup_iter + 4_500:
                    #     lambda_depth_order = 0.01
                    # if iteration > warmup_iter + 6_000:
                    #     lambda_depth_order = 0.001
                    pass
                # else:
                #     if iteration > 3_000:
                #         lambda_depth_order = 1.
                #     if iteration > 6_000:
                #         lambda_depth_order = 0.1
                #     if iteration > 9_000:
                #         lambda_depth_order = 0.01
                #     if iteration > 12_000:
                #         lambda_depth_order = 0.001
                order_supervision_depth = charts_prior_depths_up[viewpoint_idx].to(surf_depth.device)
                # if iteration < 3000 and stage == "fine":
                #     lambda_depth_order = 0.0
                if lambda_depth_order > 0:
                    depth_order_prior_loss = lambda_depth_order * compute_depth_order_loss(
                        depth=surf_depth,
                        prior_depth=order_supervision_depth.cuda(),
                        scene_extent=gaussians.spatial_lr_scale,
                        max_pixel_shift_ratio=0.05,
                        normalize_loss=True,
                        log_space=True,
                        log_scale=20.,
                        reduction="mean",
                        debug=False,
                    )

            depth_prior_loss = depth_prior_loss + depth_order_prior_loss
            total_regularization_loss = depth_prior_loss + normal_prior_loss + curv_prior_loss
            
        else:
            # dense supervision branch (与你原代码一致) - ALL GEOMETRY SUPERVISION DISABLED
            dense_depth_regularization_schedule = dense_regul
            lambda_dense_depth = 0.0  # DISABLED
            if dense_depth_regularization_schedule == "default":
                use_chart_view_every_n_iter_local = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter_local = 999_999
                    # lambda_dense_depth = 1.  # DISABLED
                if iteration > 7_000:
                    pass  # lambda_dense_depth = 0.1  # DISABLED
                if iteration > 15_000:
                    pass  # lambda_dense_depth = 0.01  # DISABLED
                if iteration > 20_000:
                    pass  # lambda_dense_depth = 0.001  # DISABLED
                if iteration > 25_000:
                    pass  # lambda_dense_depth = 0.0001  # DISABLED
            elif dense_depth_regularization_schedule == "strong":
                # lambda_dense_depth = 1.  # DISABLED
                use_chart_view_every_n_iter_local = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter_local = 999_999
                    # lambda_dense_depth = 1.  # DISABLED
            elif dense_depth_regularization_schedule == "weak":
                # lambda_dense_depth = 0.  # DISABLED
                use_chart_view_every_n_iter_local = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter_local = 15
                    # lambda_dense_depth = 0.1  # DISABLED
            elif dense_depth_regularization_schedule == "none":
                # lambda_dense_depth = 0.  # DISABLED
                use_chart_view_every_n_iter_local = 5
            else:
                raise ValueError(f"Invalid dense depth regularization schedule: {dense_depth_regularization_schedule}")

            dense_supervision_depth = getattr(dataset, "dense_depth_priors", [None])[viewpoint_idx].to(surf_depth.device) if getattr(dataset, "dense_depth_priors", None) is not None else None
            if lambda_dense_depth > 0 and dense_supervision_depth is not None:
                depth_prior_loss = lambda_dense_depth * compute_depth_order_loss(
                    depth=surf_depth,
                    prior_depth=dense_supervision_depth,
                    scene_extent=gaussians.spatial_lr_scale,
                    max_pixel_shift_ratio=0.05,
                    normalize_loss=True,
                    log_space=True,
                    log_scale=20.,
                    reduction="mean",
                    debug=False,
                )
            else:
                depth_prior_loss = torch.zeros_like(loss.detach())

            normal_prior_loss = torch.zeros_like(loss.detach())
            curv_prior_loss = torch.zeros_like(loss.detach())

            total_regularization_loss = depth_prior_loss + normal_prior_loss + curv_prior_loss
            #total_regularization_loss = normal_prior_loss + curv_prior_loss
            del prior_depth, prior_conf, prior_curv, prior_norm

        # anisotropy reg
        if lambda_anisotropy > 0.:
            gaussians_scaling = gaussians.get_scaling
            anisotropy_loss = lambda_anisotropy * (
                torch.clamp_min(gaussians_scaling.max(dim=1).values / gaussians_scaling.min(dim=1).values, anisotropy_max_ratio)
                - anisotropy_max_ratio
            ).mean()
            total_regularization_loss = total_regularization_loss + anisotropy_loss

        total_loss = total_loss + total_regularization_loss

        if stage == "fine" and dataset.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(dataset.time_smoothness_weight, dataset.l1_time_planes, dataset.plane_tv_weight)
            total_loss += tv_loss
        # backward
        total_loss.backward()

        # Gradient clipping for deformation network
        torch.nn.utils.clip_grad_norm_(gaussians._deformation.parameters(), max_norm=1.0)

        torch.cuda.empty_cache()
        #del render_pkg
        iter_end.record()
        if iteration % 100 == 0:  # 每100次迭代清理一次
            torch.cuda.empty_cache()
            gc.collect()
        # logging & EMA
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log if isinstance(dist_loss, torch.Tensor) else ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log if isinstance(normal_loss, torch.Tensor) else ema_normal_for_log
            ema_prior_depth_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_prior_depth_for_log if isinstance(depth_prior_loss, torch.Tensor) else ema_prior_depth_for_log
            ema_prior_normal_for_log = 0.4 * normal_prior_loss.item() + 0.6 * ema_prior_normal_for_log if isinstance(normal_prior_loss, torch.Tensor) else ema_prior_normal_for_log
            ema_prior_curvature_for_log = 0.4 * curv_prior_loss.item() + 0.6 * ema_prior_curvature_for_log if isinstance(curv_prior_loss, torch.Tensor) else ema_prior_curvature_for_log
            ema_prior_anisotropy_for_log = 0.4 * anisotropy_loss.item() + 0.6 * ema_prior_anisotropy_for_log if (lambda_anisotropy > 0. and isinstance(anisotropy_loss, torch.Tensor)) else ema_prior_anisotropy_for_log

            # Comprehensive Tensorboard logging
            if tb_writer is not None:
                # Stage-specific logging
                stage_prefix = f"{stage}/"
                
                # Main losses
                tb_writer.add_scalar(stage_prefix + 'train_loss_patches/total_loss', ema_loss_for_log, iteration)
                tb_writer.add_scalar(stage_prefix + 'train_loss_patches/l1_loss', Ll1.item(), iteration)
                tb_writer.add_scalar(stage_prefix + 'train_loss_patches/ssim_loss', (1.0 - ssim(image, gt_image)).item(), iteration)
                tb_writer.add_scalar(stage_prefix + 'train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar(stage_prefix + 'train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                
                # Regularization losses
                tb_writer.add_scalar(stage_prefix + 'regularization/depth_prior', ema_prior_depth_for_log, iteration)
                tb_writer.add_scalar(stage_prefix + 'regularization/normal_prior', ema_prior_normal_for_log, iteration)
                tb_writer.add_scalar(stage_prefix + 'regularization/curvature_prior', ema_prior_curvature_for_log, iteration)
                if lambda_anisotropy > 0.:
                    tb_writer.add_scalar(stage_prefix + 'regularization/anisotropy', ema_prior_anisotropy_for_log, iteration)
                
                # Scheduling parameters
                tb_writer.add_scalar(stage_prefix + 'schedule/lambda_normal', lambda_normal, iteration)
                tb_writer.add_scalar(stage_prefix + 'schedule/lambda_dist', lambda_dist, iteration)
                
                # Charts regularization scheduling (when using charts)
                if iteration % use_chart_view_every_n_iter == 0:
                    #tb_writer.add_scalar(stage_prefix + 'schedule/regularization_factor', regularization_factor, iteration)
                    tb_writer.add_scalar(stage_prefix + 'schedule/lambda_prior_depth', lambda_prior_depth, iteration)
                    tb_writer.add_scalar(stage_prefix + 'schedule/lambda_prior_depth_derivative', lambda_prior_depth_derivative, iteration)
                    tb_writer.add_scalar(stage_prefix + 'schedule/lambda_prior_normal', lambda_prior_normal, iteration)
                    tb_writer.add_scalar(stage_prefix + 'schedule/lambda_prior_curvature', lambda_prior_curvature, iteration)
                    
                    # Depth order scheduling
                    if True:  # depth order regularization
                        tb_writer.add_scalar(stage_prefix + 'schedule/lambda_depth_order', lambda_depth_order, iteration)
                
                # Dense regularization scheduling (when using dense supervision)
                else:
                    if use_dense_supervision:
                        tb_writer.add_scalar(stage_prefix + 'schedule/lambda_dense_depth', lambda_dense_depth, iteration)

                
                # Model statistics
                tb_writer.add_scalar(stage_prefix + 'model/total_points', len(gaussians.get_xyz.detach()), iteration)
                
                
                # Learning rate
                if hasattr(gaussians, 'optimizer') and gaussians.optimizer is not None:
                    for param_group in gaussians.optimizer.param_groups:
                        tb_writer.add_scalar(stage_prefix + f'lr/{param_group["name"]}', param_group['lr'], iteration)
                
                # Rendering statistics
                if 'rend_alpha' in render_pkg:
                    tb_writer.add_scalar(stage_prefix + 'render/alpha_mean', render_pkg['rend_alpha'].mean().item(), iteration)
                if 'surf_depth' in render_pkg:
                    tb_writer.add_scalar(stage_prefix + 'render/surf_depth_min', render_pkg['surf_depth'].min().item(), iteration)
                    tb_writer.add_scalar(stage_prefix + 'render/surf_depth_max', render_pkg['surf_depth'].max().item(), iteration)
                    tb_writer.add_scalar(stage_prefix + 'render/surf_depth_mean', render_pkg['surf_depth'].mean().item(), iteration)
                
                
                
                # Memory usage
                if torch.cuda.is_available():
                    tb_writer.add_scalar(stage_prefix + 'memory/gpu_memory_mb', torch.cuda.memory_allocated() / 1024 / 1024, iteration)
                    tb_writer.add_scalar(stage_prefix + 'memory/gpu_cache_mb', torch.cuda.memory_reserved() / 1024 / 1024, iteration)
                
                # Iteration timing
                tb_writer.add_scalar(stage_prefix + 'timing/iteration_time_ms', iter_start.elapsed_time(iter_end), iteration)

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz.detach())}",
                    "p_depth": f"{ema_prior_depth_for_log:.{5}f}",
                    "p_normal": f"{ema_prior_normal_for_log:.{5}f}",
                    "p_curvature": f"{ema_prior_curvature_for_log:.{5}f}"
                }
                if lambda_anisotropy > 0.:
                    loss_dict["aniso"] = f"{ema_prior_anisotropy_for_log:.{5}f}"
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            # save logs and images
            if (iteration % save_log_images_every_n_iter == 0) or (iteration == 1):
                # pick supervision images for visualization
                if iteration % use_chart_view_every_n_iter == 0:
                    supervision_depth = charts_depths_up[viewpoint_idx]
                    #print(supervision_depth.shape)
                    supervision_normal = charts_normals_up[viewpoint_idx]
                else:
                    supervision_depth = dense_supervision_depth.squeeze().unsqueeze(0) if dense_supervision_depth is not None else torch.zeros_like(charts_depths[0])
                    supervision_normal = torch.zeros_like(charts_normals[0])

                if save_log_images:
                    print("time:", viewpoint_cam.time)
                    print("image name:", viewpoint_cam.image_name)
                    figsize = 30
                    height, width = gt_image.shape[-2:]
                    nrows = 2
                    ncols = 3
                    plt.figure(figsize=(figsize, figsize * height / width * nrows / ncols))
                    plt.subplot(nrows, ncols, 1)
                    plt.title("GT Image")
                    plt.imshow(gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    plt.subplot(nrows, ncols, 2)
                    plt.title("Charts Depth")
                    plt.imshow(supervision_depth.cpu().numpy().squeeze(), cmap="Spectral")
                    plt.colorbar()
                    plt.subplot(nrows, ncols, 3)
                    plt.title("Charts Normal")
                    plt.imshow((-supervision_normal + 1).permute(1,2,0).clamp(0, 2).cpu().numpy() / 2)
                    plt.subplot(nrows, ncols, 4)
                    plt.title("Rendered Image")
                    plt.imshow(image.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    plt.subplot(nrows, ncols, 5)
                    plt.title("Rendered Depth")
                    plt.imshow(surf_depth.detach()[0].cpu().numpy(), cmap="Spectral")
                    plt.colorbar()
                    plt.subplot(nrows, ncols, 6)
                    plt.title("Rendered Normal")
                    plt.imshow((-rend_normal.detach() + 1).permute(1, 2, 0).clamp(0, 2).cpu().numpy() / 2)
                    plt.savefig(f"{dataset.model_path}/{stage}_{iteration}.png")
                    plt.close()

            # save model periodically with stage in filename
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians ({stage})")
                scene.save(iteration)

            # Test on test cameras every 500 iterations
            if iteration % 500 == 0:
                with torch.no_grad():
                    print(f"\n[ITER {iteration}] Evaluating on {len(test_cameras)} test cameras...")
                    test_cameras = scene.getTestCameras()
                    if len(test_cameras) > 0:
                        psnr_test = 0.0
                        l1_test = 0.0
                        ssim_test = 0.0

                        for idx, viewpoint in enumerate(test_cameras):
                            test_render_pkg = render(viewpoint, gaussians, pipe, background, stage='coarse' if stage == "coarse" else 'fine')
                            test_image = torch.clamp(test_render_pkg["render"], 0.0, 1.0)
                            test_gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                            l1_test += l1_loss(test_image, test_gt_image).mean().double()
                            psnr_test += psnr(test_image, test_gt_image).mean().double()

                            try:
                                ssim_val = ssim(test_image, test_gt_image).mean().double()
                                ssim_test += ssim_val
                            except:
                                ssim_test += 0.0

                        psnr_test /= len(test_cameras)
                        l1_test /= len(test_cameras)
                        ssim_test /= len(test_cameras)

                        print(f"[ITER {iteration}] Test Results - L1: {l1_test:.5f}, PSNR: {psnr_test:.4f}, SSIM: {ssim_test:.4f}")

                        if tb_writer:
                            tb_writer.add_scalar(f'{stage}/test/l1_loss', l1_test, iteration)
                            tb_writer.add_scalar(f'{stage}/test/psnr', psnr_test, iteration)
                            tb_writer.add_scalar(f'{stage}/test/ssim', ssim_test, iteration)
                    else:
                        print(f"[ITER {iteration}] No test cameras available for evaluation")

                torch.cuda.empty_cache()

            # densification/pruning/grow: keep thresholds stage-specific
            # Skip densification/pruning when Gaussian parameters are frozen
            should_skip_densification = (
                (stage == "fine" and opt.freeze_gaussians_in_fine and iteration <= warmup_iter) or
                warmup_mode  # Always skip in warmup mode
            )
            
            
            if iteration < opt.densify_until_iter and not should_skip_densification:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    # linear schedule for fine stage (as in your original implementation)
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / (opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / (opt.densify_until_iter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # note: we reuse the original densify/prune functions
                    pts_num = gaussians.get_xyz.shape[0]
                    #print("opacity threshold:", opacity_threshold)
                    if hasattr(gaussians, "densify_and_prune"):
                        gaussians.densify_and_prune(densify_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, pts_num, stage)
                    else:
                        gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)

                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(cameras=dense_viewpoint_cams if use_dense_supervision else viewpoint_cams)
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
               
            # optimizer step (do not step after final iter)
            max_iter = opt.iterations + warmup_iter if (stage == "fine" and opt.freeze_gaussians_in_fine) else opt.iterations
            if iteration < max_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint ({stage})")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt_{stage}_{iteration}.pth")

        # network GUI handling (unchanged)
        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg_c = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg_c, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

    progress_bar.close()


def training(
    dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
    use_refined_charts, use_mip_filter, dense_data_path, use_chart_view_every_n_iter,
    normal_consistency_from, distortion_from,
    depthanythingv2_checkpoint_dir, depthanything_encoder,
    dense_regul, coarse_iterations, warmup_iterations=1000, preprocessed_priors_dir=None, frames=10,
    warmup_frame_sampling="all"
):
    """
    重构后的 training：负责初始化 gaussians、charts、dense 数据等，
    然后依序调用 scene_reconstruction 两次（coarse -> fine）。
    """
    tb_writer = prepare_output_and_logger(dataset)
    #print(dataset)

    # 设置预处理priors目录
    if preprocessed_priors_dir:
        dataset.preprocessed_priors_dir = preprocessed_priors_dir
        print(f"[INFO] 使用预处理priors目录: {preprocessed_priors_dir}")
    else:
        dataset.preprocessed_priors_dir = None

    # Create gaussians from charts data (和你原脚本一致)
    gaussians = GaussianModel(dataset.sh_degree, dataset)
    # 把 gaussians 挂到 dataset，方便 scene_reconstruction 共用同一份 gaussians
    dataset._gaussians = gaussians

    # Sparse data scene
    scene = Scene(dataset, gaussians, shuffle=False)

    # Dense data processing（若指定）
    if dense_data_path is not None:
        print(f"[INFO] Loading dense supervision data from: {dense_data_path}")
        dense_dataset = copy.deepcopy(dataset)
        dense_dataset.source_path = dense_data_path
        dense_dataset.model_path = os.path.join(dataset.model_path, 'dense_data')
        os.makedirs(dense_dataset.model_path, exist_ok=True)
        dense_gaussians = GaussianModel(dataset.sh_degree)
        dense_scene = Scene(dense_dataset, dense_gaussians, shuffle=False)
        
        dense_viewpoint_cams = dense_scene.getTrainCameras()
        dataset.use_dense_supervision = True
        dataset.dense_viewpoint_cams = dense_viewpoint_cams
        dataset.dense_viewpoint_idx_stack = None
        print(f"          > Number of dense cameras: {len(dense_viewpoint_cams)}")
        print(f"          > Changing spatial lr scale from {gaussians.spatial_lr_scale} to {dense_gaussians.spatial_lr_scale}")
        gaussians.spatial_lr_scale = dense_gaussians.spatial_lr_scale
        print(f"          > Resolution of dense data: {dense_viewpoint_cams[0].original_image.shape}")
    else:
        dataset.use_dense_supervision = False
        dataset.dense_viewpoint_cams = None
        dataset.dense_viewpoint_idx_stack = None
        use_chart_view_every_n_iter = 1

    # load charts_data and build gaussian params (跟原脚本逻辑相同)
    print("[INFO] Loading charts data...")
    charts_data_path = f'{dataset.source_path}/charts_data.npz'
    charts_data = load_charts_data(charts_data_path)
    charts_data['confs'] = charts_data['confs']
    print("[WARNING] Confidence values are not being subtracted by 1.0 as in the original implementation.")
    print("Minimum confidence: ", charts_data['confs'].min())
    print("Maximum confidence: ", charts_data['confs'].max())
    print("Maximum depth: ", charts_data['depths'].max())

    h_charts, w_charts = charts_data['pts'].shape[-3:-1]
    print('='*50)
    print(h_charts, w_charts)
    print('='*50)
    train_cameras = scene.getTrainCameras()
    _images = [
        torch.nn.functional.interpolate(cam.original_image[None].cuda(), (h_charts, w_charts), mode="bilinear", antialias=True)[0].permute(1, 2, 0)
        for cam in train_cameras[::frames]
    ]
    # for cam in train_cameras[::10]:
    #     print(cam.image_name)
    print(len(_images))
    ply_path = f'{dataset.source_path}/points.ply'
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # 转成 numpy 数组
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(points)
    from utils.graphics_utils import BasicPointCloud
    bpc = BasicPointCloud(points=points, colors=colors, normals=normals)
    if True:
        gaussians.create_from_pcd(bpc, gaussians.spatial_lr_scale)
    if False:
        # Filter charts_data to only include training cameras
        train_cameras = scene.getTrainCameras()

        # Create a mapping from camera numbers to indices
        import re
        def extract_camera_number_from_path(image_path):
            """Extract camera number from image path"""
            basename = os.path.basename(image_path)
            if 'cam_' in basename:
                return int(basename[7])
            # matches = re.findall(r'cam0*(\d+)', basename, re.IGNORECASE)
            #return int(matches[0]) if matches else 0

        # Get camera numbers for all cameras in charts_data (assumes same order as total cameras available)
        avail_camera_list = [os.path.join('cam{:02d}'.format(i+1), '') for i in range(8)] # Assume 8 cameras max

        # Get actual training camera numbers from train_cameras
        train_camera_nums = set()
        for cam in train_cameras:
            print(cam.image_name)
            train_camera_nums.add(extract_camera_number_from_path(cam.image_name))

        print(f"[INFO] Filtering charts data to training cameras only: {sorted(train_camera_nums)}")
        print(f"[INFO] Original charts_data has shape pts: {charts_data['pts'].shape}, confs: {charts_data['confs'].shape}")
        print(avail_camera_list)
        print(train_camera_nums)
        # Filter charts_data based on training camera numbers
        # charts_data dimensions are typically: (n_cameras, height, width, channels)
        filtered_indices = []
        for i in range(len(avail_camera_list)):
            if (i+1) in train_camera_nums:  # Camera numbers start from 1
                filtered_indices.append(i)

        if len(filtered_indices) == 0:
            print("[WARNING] No training cameras found in charts data, using original data")
            filtered_charts_data = charts_data
        else:
            # Filter tensors
            filtered_charts_data = {}
            for key in charts_data.keys():
                if key != 'scale_factor' and key in charts_data:
                    # Check if we need to filter this key based on camera dimension
                    tensor = charts_data[key]
                    if tensor.dim() >= 1 and tensor.shape[0] >= len(avail_camera_list):
                        # Filter along the first dimension (camera dimension)
                        filtered_charts_data[key] = tensor[filtered_indices]
                        print(f"[INFO] Filtered {key}: {tensor.shape} -> {filtered_charts_data[key].shape}")
                    else:
                        # Copy as-is if doesn't have camera dimension
                        filtered_charts_data[key] = tensor
                        print(f"[INFO] Kept {key} unchanged: {tensor.shape}")
                else:
                    # Keep scale_factor as-is
                    filtered_charts_data[key] = charts_data[key]

            filtered_charts_data['scale_factor'] = charts_data['scale_factor']

        print(f"[INFO] Final filtered charts_data: pts shape {filtered_charts_data['pts'].shape}, num training cameras: {len(filtered_indices)}")

        gaussian_params = get_gaussian_parameters_from_charts_data(
            charts_data=charts_data,  # Use filtered data
            images=_images,
            conf_th=-1.,  # TODO
            ratio_th=5.,
            normal_scale=1e-10,
            normalized_scales=0.5,
        )

        n_max_gaussians = 200000
        if n_max_gaussians > -1 and n_max_gaussians < len(gaussian_params['means']):
            downsample_factor = len(gaussian_params['means']) / n_max_gaussians
            sample_idx = torch.randperm(len(gaussian_params['means']))[:n_max_gaussians]
        else:
            downsample_factor = 1.0
            sample_idx = torch.arange(len(gaussian_params['means']))

        _means = gaussian_params['means'][sample_idx]
        _scales = gaussian_params['scales'][..., :2][sample_idx] * downsample_factor
        _quaternions = gaussian_params['quaternions'][sample_idx]
        _colors = gaussian_params['colors'][sample_idx]
        print('='*50)
        print(_scales.max(), _scales.min())
        print('='*50)
        if dataset.use_dense_supervision:
            with torch.no_grad():
                _means = torch.cat([_means, dense_gaussians.get_xyz.detach()], dim=0)
                _scales = torch.cat([_scales, dense_gaussians.get_scaling.detach()], dim=0)
                _quaternions = torch.cat([_quaternions, dense_gaussians.get_rotation.detach()], dim=0)
                _colors = torch.cat([_colors, SH2RGB(dense_gaussians._features_dc.detach()[:, 0])], dim=0)
        print(_means.max(), _means.min())
        print(_means.shape)
        gaussians.create_from_parameters(_means, _scales, _quaternions, _colors, gaussians.spatial_lr_scale)
        gaussians.save_ply('./initial_gs.ply')
    print("[INFO] Gaussians created from charts data.")

    # build priors for charts and store to dataset for use in scene_reconstruction
    #charts_priors = build_priors_from_charts_data(charts_data, scene.getTrainCameras())
    #dataset.charts_priors = charts_priors

    # optionally prepare dense depth priors using Depth-Anything V2 (与原脚本保持一致)
    if dataset.use_dense_supervision:
        print("[INFO] Building depth priors from dense data...")
        from matcha.pointmap.depthanythingv2 import load_model as load_depthanythingv2
        from matcha.pointmap.depthanythingv2 import apply_depthanything

        dav2 = load_depthanythingv2(checkpoint_dir=depthanythingv2_checkpoint_dir, encoder=depthanything_encoder, device='cuda')
        dav2.eval()
        dense_depth_priors = []
        with torch.no_grad():
            for i_image in range(len(dataset.dense_viewpoint_cams)):
                gt_image = dataset.dense_viewpoint_cams[i_image].original_image.permute(1, 2, 0)
                supervision_disparity = apply_depthanything(dav2, image=gt_image)
                supervision_disparity = (supervision_disparity - supervision_disparity.min()) / (supervision_disparity.max() - supervision_disparity.min())
                supervision_depth = 1. / (0.1 + 0.9 * supervision_disparity)
                dense_depth_priors.append(supervision_depth.squeeze().unsqueeze(0).to('cpu'))
        del dav2
        gc.collect()
        torch.cuda.empty_cache()
        dataset.dense_depth_priors = dense_depth_priors
        print("[INFO] Depth priors for dense data built.")
    tb_writer = prepare_output_and_logger(dataset)
    #gaussians = GaussianModel(dataset.sh_degree, opt) 
    #scene = Scene(dataset, gaussians, shuffle=False)
    # Log initialization information
    print("saving initial state")
    scene.save(0)
    # call coarse stage first (如果 coarse_iterations > 0)
    if coarse_iterations and coarse_iterations > 0:
        print("[INFO] Starting COARSE stage")
        if tb_writer:
            tb_writer.add_text('stage_transition', "Starting COARSE stage", 0)
            tb_writer.add_scalar('stage/current', 0, 0)  # 0 for coarse
        scene_reconstruction(
            dataset=dataset,
            opt=opt,
            pipe=pipe,
            testing_iterations=testing_iterations,
            saving_iterations=saving_iterations,
            checkpoint_iterations=checkpoint_iterations,
            checkpoint=checkpoint,
            use_refined_charts=use_refined_charts,
            use_mip_filter=use_mip_filter,
            dense_data_path=dense_data_path,
            use_chart_view_every_n_iter=use_chart_view_every_n_iter,
            normal_consistency_from=normal_consistency_from,
            distortion_from=distortion_from,
            depthanythingv2_checkpoint_dir=depthanythingv2_checkpoint_dir,
            depthanything_encoder=depthanything_encoder,
            dense_regul=dense_regul,
            stage="coarse",
            train_iter=coarse_iterations,
            gaussians=gaussians,
            scene=scene,
            tb_writer=tb_writer,
            frames=frames
        )

    # warm-up stage: train only deformation MLP, freeze gaussian parameters
    if warmup_iterations and warmup_iterations > 0:
        print(f"[INFO] Starting WARM-UP stage ({warmup_iterations} iterations)")
        print("[INFO] Warm-up mode: Gaussian parameters frozen, only training deformation MLP")
        print("[INFO] Warm-up mode: No densify/prune/gene operations")
        if tb_writer:
            tb_writer.add_text('stage_transition', f"Starting WARM-UP stage ({warmup_iterations} iterations)", coarse_iterations if coarse_iterations else 0)
            tb_writer.add_scalar('stage/current', 1, coarse_iterations if coarse_iterations else 0)  # 1 for warmup
        scene_reconstruction(
            dataset=dataset,
            opt=opt,
            pipe=pipe,
            testing_iterations=testing_iterations,
            saving_iterations=saving_iterations,
            checkpoint_iterations=checkpoint_iterations,
            checkpoint=None,  # start from current in-memory gaussians
            use_refined_charts=use_refined_charts,
            use_mip_filter=use_mip_filter,
            dense_data_path=dense_data_path,
            use_chart_view_every_n_iter=use_chart_view_every_n_iter,
            normal_consistency_from=normal_consistency_from,
            distortion_from=distortion_from,
            depthanythingv2_checkpoint_dir=depthanythingv2_checkpoint_dir,
            depthanything_encoder=depthanything_encoder,
            dense_regul=dense_regul,
            stage="warmup",
            train_iter=warmup_iterations,
            gaussians=gaussians,
            scene=scene,
            tb_writer=tb_writer,
            frames=frames,
            warmup_mode=True,  # Key: enable warmup mode
            warmup_frame_sampling=warmup_frame_sampling  # Frame sampling mode for warmup
        )

    # then fine stage
    print("[INFO] Starting FINE stage")
    print("[INFO] training iterations for fine stage: ", opt.iterations)
    # Calculate global iteration before fine stage
    global_iter_before_fine = (coarse_iterations if coarse_iterations else 0) + (warmup_iterations if warmup_iterations else 0)
    if tb_writer:
        tb_writer.add_text('stage_transition', "Starting FINE stage", global_iter_before_fine)
        tb_writer.add_scalar('stage/current', 2, global_iter_before_fine)  # 2 for fine (0=coarse, 1=warmup, 2=fine)
    scene_reconstruction(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        testing_iterations=testing_iterations,
        saving_iterations=saving_iterations,
        checkpoint_iterations=checkpoint_iterations,
        checkpoint=None,  # typically start fine from the current in-memory gaussians (not from external checkpoint)
        use_refined_charts=use_refined_charts,
        use_mip_filter=use_mip_filter,
        dense_data_path=dense_data_path,
        use_chart_view_every_n_iter=use_chart_view_every_n_iter,
        normal_consistency_from=normal_consistency_from,
        distortion_from=distortion_from,
        depthanythingv2_checkpoint_dir=depthanythingv2_checkpoint_dir,
        depthanything_encoder=depthanything_encoder,
        dense_regul=dense_regul,
        stage="fine",
        train_iter=opt.iterations,
        gaussians=gaussians,
        scene=scene,
        tb_writer=tb_writer,
        frames=frames
    )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        # Additional comprehensive metrics
        tb_writer.add_scalar('training/iteration', iteration, iteration)
        tb_writer.add_scalar('training/elapsed_time_ms', elapsed, iteration)
        
        # Model statistics
        tb_writer.add_scalar('model/num_gaussians', scene.gaussians.get_xyz.shape[0], iteration)
        
        # Gaussian properties statistics
        if hasattr(scene.gaussians, 'get_scaling'):
            scales = scene.gaussians.get_scaling
            tb_writer.add_scalar('model/avg_scale', scales.mean().item(), iteration)
            tb_writer.add_scalar('model/min_scale', scales.min().item(), iteration)
            tb_writer.add_scalar('model/max_scale', scales.max().item(), iteration)
        
        if hasattr(scene.gaussians, 'get_rotation'):
            rotations = scene.gaussians.get_rotation
            tb_writer.add_scalar('model/rotation_magnitude', rotations.norm(dim=1).mean().item(), iteration)
        
        if hasattr(scene.gaussians, 'get_opacity'):
            opacities = scene.gaussians.get_opacity
            tb_writer.add_scalar('model/avg_opacity', opacities.mean().item(), iteration)
            tb_writer.add_scalar('model/min_opacity', opacities.min().item(), iteration)
            tb_writer.add_scalar('model/max_opacity', opacities.max().item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                            
                            # Additional rendering metrics
                            if 'surf_depth' in render_pkg:
                                surf_depth = render_pkg['surf_depth']
                                tb_writer.add_scalar(config['name'] + f"_view_{viewpoint.image_name}/surf_depth_min", surf_depth.min().item(), iteration)
                                tb_writer.add_scalar(config['name'] + f"_view_{viewpoint.image_name}/surf_depth_max", surf_depth.max().item(), iteration)
                                tb_writer.add_scalar(config['name'] + f"_view_{viewpoint.image_name}/surf_depth_mean", surf_depth.mean().item(), iteration)
                            
                            if 'rend_normal' in render_pkg:
                                rend_normal_mag = render_pkg['rend_normal'].norm(dim=0).mean().item()
                                tb_writer.add_scalar(config['name'] + f"_view_{viewpoint.image_name}/rend_normal_magnitude", rend_normal_mag, iteration)
                            
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    # Calculate metrics
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    # Calculate SSIM if available
                    try:
                        ssim_val = ssim(image, gt_image).mean().double()
                        ssim_test += ssim_val
                    except:
                        ssim_test += 0.0

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    
                    # Additional validation metrics
                    tb_writer.add_scalar(config['name'] + '/validation/num_cameras', len(config['cameras']), iteration)
                    tb_writer.add_scalar(config['name'] + '/validation/iteration', iteration, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=None)  # 6009
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_500, 5_000, 7_000, 8_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_refined_charts", action="store_true", default=False)
    parser.add_argument("--use_mip_filter", action="store_true", default=False)
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument("--use_chart_view_every_n_iter", type=int, default=999_999)
    parser.add_argument("--normal_consistency_from", type=int, default=3500)
    parser.add_argument("--distortion_from", type=int, default=1500)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='../Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default', help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')
    parser.add_argument('--preprocessed_priors_dir', type=str, default=None, help='Directory containing preprocessed priors (depths, confs, normals, curvs)')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--seed', type=int, default=10086, help='Random seed for reproducible results. If not specified, uses system time.')
    #parser.add_argument('--coarse_iterations', type=int, default=0, help='Number of static coarse training iterations')
    parser.add_argument('--warmup_iterations', type=int, default=0,
                        help='Number of warm-up iterations (before fine stage, only train deformation MLP, freeze gaussians)')
    parser.add_argument('--warmup_frame_sampling', type=str, default='first_10_percent',
                        choices=['all', 'first_10_percent', 'uniform_every_10'],
                        help='Frame sampling mode for warmup stage: "all" (use all frames), "first_10_percent" (use first 10%% frames), "uniform_every_10" (sample every 10th frame)')
    args = parser.parse_args(sys.argv[1:])
    print("iterations for training: ", args.iterations)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Set random seed for reproducible results
    if args.seed is not None:
        set_random_seed(args.seed)
    else:
        # If no seed specified, use system time but still set it for consistency
        import time
        default_seed = int(time.time()) % (2**32)
        print(f"[INFO] No seed specified, using system time as seed: {default_seed}")
        set_random_seed(default_seed)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.port is None:
        import time
        current_time = time.strftime("%H%M%S", time.localtime())[2:]
        args.port = int(current_time)
        print(f"Randomly selected port: {args.port}")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 调用 training（coarse -> fine）
    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.use_refined_charts, args.use_mip_filter,
        args.dense_data_path, args.use_chart_view_every_n_iter,
        args.normal_consistency_from, args.distortion_from,
        args.depthanythingv2_checkpoint_dir, args.depthanything_encoder,
        args.dense_regul, args.coarse_iterations, args.warmup_iterations,
        args.preprocessed_priors_dir, args.frames, args.warmup_frame_sampling
    )

    print("\nTraining complete.")
