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
from utils.render_utils import create_videos
from scene.cameras import Camera
import copy

import open3d as o3d
import numpy as np


def _as_4x4_w2c(w2c):
    if w2c.shape == (3, 4):
        return np.vstack([w2c, [0, 0, 0, 1]])
    return w2c.copy()


def _sort_cameras_for_loop(w2c_list):
    centers = np.array([np.linalg.inv(w2c)[:3, 3] for w2c in w2c_list])
    origin = np.median(centers, axis=0)
    centered = centers - origin

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[0] < 1e-8 or singular_values[1] / singular_values[0] < 1e-3:
        return w2c_list

    coords = centered @ vh[:2].T
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    order = np.argsort(angles)
    return [w2c_list[i] for i in order]


def gen_path(w2c_list, num_views, closed_loop=True, center_shrink=0.9, sort_closed_loop=True):
    """
    Generate interpolated camera path between adjacent cameras.

    Args:
        w2c_list: List of world-to-camera matrices (4x4 or 3x4)
        num_views: Total number of views to generate
        closed_loop: If True, connect last camera back to first
        center_shrink: Move camera centers toward the median training-camera
            center. Values below 1.0 make indoor paths less likely to leave
            the reconstructed room volume.
        sort_closed_loop: Sort cameras by their spatial angle before closing
            the path. This avoids connecting arbitrary dataset order jumps.

    Returns:
        List of interpolated w2c matrices
    """
    from scipy.spatial.transform import Rotation, Slerp

    # Ensure all w2c are 4x4.
    w2c_4x4 = [_as_4x4_w2c(w2c) for w2c in w2c_list]

    num_cameras = len(w2c_4x4)
    if num_cameras < 2:
        return w2c_4x4 * num_views if num_cameras == 1 else []

    if closed_loop and sort_closed_loop:
        w2c_4x4 = _sort_cameras_for_loop(w2c_4x4)

    camera_centers = np.array([np.linalg.inv(w2c)[:3, 3] for w2c in w2c_4x4])
    path_center = np.median(camera_centers, axis=0)
    camera_centers = path_center + center_shrink * (camera_centers - path_center)

    # Calculate segments
    if closed_loop:
        num_segments = num_cameras
    else:
        num_segments = num_cameras - 1

    views_per_segment = num_views // num_segments
    extra_views = num_views % num_segments

    interpolated_w2c = []

    for i in range(num_segments):
        cam_idx1 = i % num_cameras
        cam_idx2 = (i + 1) % num_cameras

        w2c1 = w2c_4x4[cam_idx1]
        w2c2 = w2c_4x4[cam_idx2]

        # Decompose to rotation and camera center. Interpolating the w2c
        # translation directly is not equivalent to interpolating camera
        # positions and can push generated viewpoints outside indoor captures.
        R1, c1 = w2c1[:3, :3], camera_centers[cam_idx1]
        R2, c2 = w2c2[:3, :3], camera_centers[cam_idx2]

        # Determine number of steps for this segment
        steps = views_per_segment + (1 if i < extra_views else 0)

        for j in range(steps):
            alpha = j / steps if steps > 0 else 0
            # Interpolate rotation using SLERP
            rot1 = Rotation.from_matrix(R1)
            rot2 = Rotation.from_matrix(R2)
            key_rots = Rotation.concatenate([rot1, rot2])
            slerp = Slerp([0, 1], key_rots)
            rot_interp = slerp(alpha)
            R_interp = rot_interp.as_matrix()

            # Interpolate camera center in world space, then rebuild w2c.
            c_interp = (1 - alpha) * c1 + alpha * c2
            t_interp = -R_interp @ c_interp

            # Construct interpolated w2c
            w2c_interp = np.eye(4)
            w2c_interp[:3, :3] = R_interp
            w2c_interp[:3, 3] = t_interp
            interpolated_w2c.append(w2c_interp)

    return interpolated_w2c[:num_views]


def build_camera_matrices(w2c, K, H, W):
    """
    Build camera matrices from w2c, intrinsics, and image dimensions.

    Args:
        w2c: World-to-camera matrix (4x4)
        K: Camera intrinsics matrix (3x3)
        H: Image height
        W: Image width

    Returns:
        Dictionary containing camera matrices
    """
    # Calculate FOV from intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    FoVx = 2 * np.arctan(W / (2 * fx))
    FoVy = 2 * np.arctan(H / (2 * fy))
    # Convert w2c to torch tensor
    world_view_transform = torch.from_numpy(w2c.T).float().cuda()

    # Build projection matrix
    from utils.graphics_utils import getProjectionMatrix
    znear = 0.01
    zfar = 100.0
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()

    # Calculate full projection transform
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    # Calculate camera center (inverse of w2c gives c2w, camera center is translation part)
    # Use robust inverse to handle potential singular matrices
    try:
        c2w = np.linalg.inv(w2c)
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse or construct from rotation transpose
        # For camera matrices: c2w = [[R^T, -R^T @ t], [0, 1]] where w2c = [[R, t], [0, 1]]
        c2w = np.eye(4, dtype=w2c.dtype)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        # Use rotation transpose (orthonormal matrix property: R^-1 = R^T)
        R_T = R.T
        c2w[:3, :3] = R_T
        c2w[:3, 3] = -R_T @ t

    camera_center = torch.from_numpy(c2w[:3, 3]).float().cuda()

    return {
        'world_view_transform': world_view_transform,
        'projection_matrix': projection_matrix,
        'full_proj_transform': full_proj_transform,
        'camera_center': camera_center,
        'image_height': H,
        'image_width': W,
        'FoVx': FoVx,
        'FoVy': FoVy,
        'w2c': w2c,
        'c2w': c2w,
    }


def generate_path_from_cameras(viewpoint_cameras, n_frames=240, closed_loop=True, path_shrink=0.9, sort_closed_loop=True):
    """
    Generate interpolated camera path between adjacent training cameras.

    Args:
        viewpoint_cameras: List of Camera objects
        n_frames: Number of frames to generate
        closed_loop: Whether to create a closed loop path
        path_shrink: Scale camera centers toward the median camera center
        sort_closed_loop: Sort cameras spatially before connecting the loop

    Returns:
        List of Camera objects for the trajectory
    """
    if len(viewpoint_cameras) == 0:
        return []

    # Extract w2c matrices from all cameras
    w2c_list = []
    for cam in viewpoint_cameras:
        if cam.time == 0.0:
            w2c = np.asarray(cam.world_view_transform.T.cpu().numpy())
            w2c_list.append(w2c)

    # Generate interpolated path
    free_view_w2c = gen_path(
        w2c_list,
        n_frames,
        closed_loop=closed_loop,
        center_shrink=path_shrink,
        sort_closed_loop=sort_closed_loop,
    )

    # Use first camera's intrinsics for all views
    first_cam = viewpoint_cameras[0]

    # Extract intrinsics from first camera
    # K matrix from FoV and image dimensions
    fx = first_cam.image_width / (2.0 * np.tan(first_cam.FoVx / 2.0))
    fy = first_cam.image_height / (2.0 * np.tan(first_cam.FoVy / 2.0))
    cx = first_cam.image_width / 2.0
    cy = first_cam.image_height / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    H, W = first_cam.image_height, first_cam.image_width

    # Build trajectory cameras
    traj = []
    for idx, w2c in enumerate(free_view_w2c):
        cam = copy.deepcopy(first_cam)
        cam.image_height = int(H / 2) * 2
        cam.image_width = int(W / 2) * 2

        # Build camera matrices
        cam_dict = build_camera_matrices(w2c, K, cam.image_height, cam.image_width)

        cam.world_view_transform = cam_dict['world_view_transform']
        cam.full_proj_transform = cam_dict['full_proj_transform']
        cam.camera_center = cam_dict['camera_center']
        cam.time = idx / n_frames

        traj.append(cam)

    loop_str = "closed loop" if closed_loop else "open path"
    print(f"[FreeView] Generated {len(traj)} camera poses ({loop_str}, shrink={path_shrink}, sort={sort_closed_loop})")

    return traj

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--closed_loop", action="store_true", help='Video path: connect last camera back to first')
    parser.add_argument("--path_shrink", default=0.9, type=float, help='Video path: move camera centers toward the median center; use 1.0 to disable')
    parser.add_argument("--no_path_sort", action="store_true", help='Video path: keep dataset camera order instead of spatial loop order')
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, opt, iteration, pipe = model.extract(args), optimization.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_frames = 240
        closed_loop = getattr(args, 'closed_loop', True)
        cam_traj = generate_path_from_cameras(
            scene.getTrainCameras(),
            n_frames=n_frames,
            closed_loop=closed_loop,
            path_shrink=args.path_shrink,
            sort_closed_loop=not args.no_path_sort,
        )
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir,
                    out_name='render_traj',
                    num_frames=n_frames)

    # if not args.skip_mesh:
    #     print("export mesh ...")
    #     os.makedirs(train_dir, exist_ok=True)
    #     # set the active_sh to 0 to export only diffuse texture
    #     gaussExtractor.gaussians.active_sh_degree = 0
    #     gaussExtractor.reconstruction(scene.getTrainCameras())
    #     # extract the mesh and save
    #     if args.unbounded:
    #         name = 'fuse_unbounded.ply'
    #         mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
    #     else:
    #         name = 'fuse.ply'
    #         depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
    #         voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
    #         sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
    #         mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
    #     o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
    #     print("mesh saved at {}".format(os.path.join(train_dir, name)))
    #     # post-process the mesh and save, saving the largest N clusters
    #     mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    #     o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
    #     print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
