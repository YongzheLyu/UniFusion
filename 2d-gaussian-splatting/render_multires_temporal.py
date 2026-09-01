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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, post_process_mesh

import open3d as o3d
import numpy as np
import re


class TemporalGaussianExtractor(GaussianExtractor):
    """
    Extended GaussianExtractor to support Deformable 2DGS temporal rendering.
    Allows extracting mesh at specific timestamps by fixing the time parameter
    while using all training camera viewpoints.
    """

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, fixed_time=None, debug_dir=None):
        """
        Reconstruct radiance field given cameras with optional fixed timestamp.

        Args:
            viewpoint_stack: List of camera viewpoints to render from
            fixed_time: If specified, all cameras will render at this timestamp
                       (for extracting mesh at a specific moment in time)
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack

        if debug_dir is not None:
            import torchvision
            os.makedirs(debug_dir, exist_ok=True)

        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            # Save original time and set fixed time if specified
            original_time = viewpoint_cam.time
            if fixed_time is not None:
                viewpoint_cam.time = fixed_time

            try:
                # Use 'fine' stage to trigger deformation MLP
                render_pkg = self.render(viewpoint_cam, self.gaussians, stage='fine')
                rgb = render_pkg['render']
                depth = render_pkg['surf_depth']
                self.rgbmaps.append(rgb.cpu())
                self.depthmaps.append(depth.cpu())

                if debug_dir is not None:
                    cam_name = getattr(viewpoint_cam, 'image_name', f'cam_{i:04d}')
                    cam_name = os.path.basename(str(cam_name)).replace(os.sep, "_")
                    timestamp_str = f"{float(viewpoint_cam.time):.6f}".replace(".", "_")
                    torchvision.utils.save_image(rgb, os.path.join(debug_dir, f"{cam_name}_t{timestamp_str}_rgb.png"))
                    depth_vis = depth / (depth.max() + 1e-8)
                    torchvision.utils.save_image(depth_vis, os.path.join(debug_dir, f"{cam_name}_t{timestamp_str}_depth.png"))
            finally:
                # Restore original time regardless of success/failure
                viewpoint_cam.time = original_time

        self.estimate_bounding_sphere()


def get_unique_timestamps(cameras):
    """Extract unique timestamps from camera list, sorted in ascending order."""
    timestamps = sorted(set([float(cam.time) for cam in cameras]))
    return timestamps


def get_camera_identity(camera):
    image_name = str(getattr(camera, "image_name", ""))
    match = re.match(r"^(cam_\d+)_\d+$", image_name)
    if match is not None:
        return match.group(1)

    R = np.asarray(camera.R).round(decimals=6)
    T = np.asarray(camera.T).round(decimals=6)
    return tuple(np.concatenate([R.reshape(-1), T.reshape(-1)]).tolist())


def select_integration_cameras(cameras, timestamp, mode="timestamp", tolerance=1e-6):
    """
    Select camera poses used by TSDF integration.

    timestamp: use cameras whose dataset timestamp matches the target timestamp.
    zero: use cameras at t=0, rendered with the target timestamp.
    unique: keep one camera for each physical/viewpoint id, rendered with the target timestamp.
    all: use all train cameras, rendered with the target timestamp.
    """
    if mode == "all":
        selected = list(cameras)
    elif mode == "zero":
        selected = [cam for cam in cameras if abs(float(cam.time)) <= tolerance]
    elif mode == "timestamp":
        selected = [cam for cam in cameras if abs(float(cam.time) - timestamp) <= tolerance]
    elif mode == "unique":
        selected = []
    else:
        raise ValueError(f"Unknown camera selection mode: {mode}")

    if mode == "unique" or len(selected) == 0:
        seen = set()
        selected = []
        for cam in cameras:
            key = get_camera_identity(cam)
            if key in seen:
                continue
            seen.add(key)
            selected.append(cam)

    return selected


def mesh_stats(mesh):
    return len(mesh.vertices), len(mesh.triangles)


def extract_mesh_at_timestamp(args, timestamp, gaussians, scene, pipe, bg_color, output_dir):
    """
    Extract multi-resolution TSDF mesh at a specific timestamp.

    Args:
        args: Command line arguments
        timestamp: Target timestamp for mesh extraction
        gaussians: GaussianModel instance
        scene: Scene instance
        pipe: PipelineParams
        bg_color: Background color tensor
        output_dir: Output directory for saving meshes

    Returns:
        Path to saved mesh file or None if extraction failed
    """
    print(f"\n{'='*70}")
    print(f"Extracting mesh at timestamp: {timestamp:.6f}")
    print(f"{'='*70}")

    train_cameras_all = scene.getTrainCameras()
    train_cameras = select_integration_cameras(
        train_cameras_all,
        timestamp,
        mode=args.camera_selection,
        tolerance=args.timestamp_tolerance,
    )
    if len(train_cameras) == 0:
        raise RuntimeError("No cameras available for TSDF integration.")
    print(
        f"Using {len(train_cameras)} integration cameras "
        f"(selection={args.camera_selection}, train total={len(train_cameras_all)})."
    )

    # Create extractor
    gaussExtractor = TemporalGaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    # Perform reconstruction at fixed timestamp
    debug_dir = None
    if args.save_debug_renders:
        timestamp_str = f"{timestamp:.6f}".replace(".", "_")
        debug_dir = os.path.join(output_dir, "debug_renders", f"t{timestamp_str}")
    gaussExtractor.reconstruction(train_cameras, fixed_time=timestamp, debug_dir=debug_dir)

    # Set active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0

    # Multi-resolution mesh extraction
    multires_factors = args.multires_factors
    meshes = []
    depth_truncs = []

    print(f"\nStarting multi-resolution extraction with factors: {multires_factors}")
    for factor in multires_factors:
        print(f'\nExtracting mesh with factor {factor}...')
        depth_trunc = gaussExtractor.radius * factor
        voxel_size = depth_trunc / args.mesh_res
        sdf_trunc = 5.0 * voxel_size

        mesh = gaussExtractor.extract_mesh_bounded(
            voxel_size=voxel_size,
            sdf_trunc=sdf_trunc,
            depth_trunc=depth_trunc
        )
        n_verts, n_faces = mesh_stats(mesh)
        if n_verts == 0 or n_faces == 0:
            print(f"  Warning: empty mesh for factor {factor}; skipping this resolution.")
            continue
        meshes.append(mesh)
        depth_truncs.append(depth_trunc)
        print(
            f'  Mesh extracted: depth_trunc={depth_trunc:.4f}, '
            f'voxel_size={voxel_size:.6f}, vertices={n_verts}, faces={n_faces}'
        )

    if len(meshes) == 0:
        print("Warning: TSDF integration produced no non-empty meshes.")
        return None

    # Build camera wrapper for multi-resolution mesh merging
    from matcha.dm_scene.cameras import CamerasWrapper, GSCamera

    gs_cameras = []
    for cam in train_cameras:
        gs_cameras.append(GSCamera(
            colmap_id=cam.colmap_id,
            R=cam.R,
            T=cam.T,
            FoVx=cam.FoVx,
            FoVy=cam.FoVy,
            image=cam.original_image,
            gt_alpha_mask=cam.gt_alpha_mask,
            image_name=cam.image_name,
            uid=cam.uid,
            data_device=cam.data_device,
            image_height=cam.image_height,
            image_width=cam.image_width,
        ))
    cameras_wrapper = CamerasWrapper(gs_cameras)

    # Merge multi-resolution meshes
    from matcha.dm_scene.meshes import Meshes, TexturesVertex, remove_faces_from_single_mesh, join_meshes_as_scene

    p3d_meshes = []
    device = 'cuda'

    print("\n=== Merging multi-resolution meshes ===")
    for i_mesh, (depth_trunc, mesh) in enumerate(zip(depth_truncs, meshes)):
        print(f"Processing mesh {i_mesh+1}/{len(meshes)} with depth_trunc={depth_trunc:.4f}...")

        verts = torch.from_numpy(np.asarray(mesh.vertices)).float().to(device)
        faces = torch.from_numpy(np.asarray(mesh.triangles)).long().to(device)
        vert_colors = torch.from_numpy(np.asarray(mesh.vertex_colors)).float().to(device)

        p3d_mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex([vert_colors]),
        )
        empty_mesh = False

        # Remove faces from lower resolutions that are covered by higher resolutions
        if i_mesh > 0:
            # Project vertices to camera views
            projections = cameras_wrapper.project_points(verts.view(1, -1, 3))
            height, width = cameras_wrapper.gs_cameras[0].image_height, cameras_wrapper.gs_cameras[0].image_width
            factors = torch.tensor([[[-width / min(height, width), -height / min(height, width)]]], device=device)
            projections = projections / factors
            visible_mask = (projections[..., 0] > -1.0) & (projections[..., 0] < 1.0) & \
                          (projections[..., 1] > -1.0) & (projections[..., 1] < 1.0)

            # Check which vertices are close to camera (within previous depth truncation)
            depths = cameras_wrapper.transform_points_world_to_view(verts.view(1, -1, 3))[..., 2]
            close_verts = depths < depth_truncs[i_mesh - 1]

            # Mark non-valid vertices and faces
            non_valid_verts = (visible_mask & close_verts).any(dim=0)
            non_valid_faces = non_valid_verts[faces].all(dim=-1)

            # Remove the corresponding faces
            try:
                p3d_mesh = remove_faces_from_single_mesh(p3d_mesh, faces_to_keep_mask=(~non_valid_faces))
            except Exception as e:
                print(f"  Warning: Error removing faces for mesh {i_mesh}: {e}")
                empty_mesh = True

        if not empty_mesh:
            p3d_meshes.append(p3d_mesh)

    # Save merged mesh
    if len(p3d_meshes) > 0:
        print(f"\nJoining {len(p3d_meshes)} meshes...")
        full_mesh = join_meshes_as_scene(p3d_meshes)
        verts = full_mesh.verts_packed()
        faces = full_mesh.faces_packed()
        vert_colors = full_mesh.textures.verts_features_packed()

        # Convert to Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
        mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors.cpu().numpy())

        # Save mesh with timestamp in filename
        timestamp_str = f"{timestamp:.4f}".replace('.', '_')
        name = f'multires_tsdf_t{timestamp_str}.ply'
        mesh_path = os.path.join(output_dir, name)
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"\nMesh saved at: {mesh_path}")
        print(f"  - Vertices: {len(mesh.vertices)}")
        print(f"  - Faces: {len(mesh.triangles)}")

        if not args.skip_post_process:
            print("\nPost-processing mesh...")
            try:
                mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
                post_path = os.path.join(output_dir, name.replace('.ply', '_post.ply'))
                o3d.io.write_triangle_mesh(post_path, mesh_post)
                print(f"Post-processed mesh saved at: {post_path}")
                print(f"  - Vertices after post-process: {len(mesh_post.vertices)}")
                print(f"  - Faces after post-process: {len(mesh_post.triangles)}")
            except Exception as e:
                print(f"Warning: post-processing failed, raw mesh is still saved. Error: {e}")

        return mesh_path
    else:
        print("Warning: No valid meshes to save!")
        return None


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Extract TSDF mesh from Deformable 2DGS at specific timestamps")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int,
                       help="Iteration to load, -1 for latest")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output")
    parser.add_argument("--voxel_size", default=-1.0, type=float,
                       help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float,
                       help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float,
                       help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int,
                       help='Mesh: number of connected clusters to export')
    parser.add_argument("--mesh_res", default=1024, type=int,
                       help='Mesh: resolution for mesh extraction')
    parser.add_argument("--multires_factors", default=[2, 8, 16], nargs='+', type=int,
                       help='Mesh: multi-resolution factors')
    parser.add_argument("--output_dir", type=str, default=None,
                       help='Path to save the output meshes')
    parser.add_argument("--skip_post_process", action="store_true",
                       help="Skip connected-component mesh post-processing")
    parser.add_argument("--save_debug_renders", action="store_true",
                       help="Save RGB/depth renders used for TSDF integration")

    # Temporal-specific arguments
    parser.add_argument("--timestamp", default=None, type=float,
                       help="Specific timestamp to extract mesh (default: use first camera's time)")
    parser.add_argument("--extract_all_timestamps", action="store_true",
                       help="Extract mesh for all unique timestamps in the dataset")
    parser.add_argument("--timestamp_list", default=None, nargs='+', type=float,
                       help="List of specific timestamps to extract (e.g., --timestamp_list 0.0 0.5 1.0)")
    parser.add_argument("--camera_selection", default="timestamp",
                       choices=["timestamp", "zero", "unique", "all"],
                       help="Camera poses used for TSDF integration. If timestamp/zero has no match, unique is used as fallback.")
    parser.add_argument("--timestamp_tolerance", default=1e-6, type=float,
                       help="Tolerance when matching camera timestamps")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Load model
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

    # Set up output directory
    output_dir = getattr(args, 'output_dir', None)
    if output_dir is None:
        output_dir = os.path.join(args.model_path, 'meshes_temporal', f"ours_{scene.loaded_iter}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get training cameras
    train_cameras = scene.getTrainCameras()
    print(f"Number of training cameras: {len(train_cameras)}")

    # Determine which timestamps to process
    if args.timestamp is not None:
        # Single specific timestamp
        timestamps = [args.timestamp]
        print(f"Extracting mesh at specified timestamp: {args.timestamp}")
    elif args.timestamp_list is not None:
        # List of timestamps
        timestamps = args.timestamp_list
        print(f"Extracting meshes at {len(timestamps)} specified timestamps: {timestamps}")
    elif args.extract_all_timestamps:
        # All unique timestamps from dataset
        timestamps = get_unique_timestamps(train_cameras)
        print(f"Found {len(timestamps)} unique timestamps in dataset")
        print(f"Timestamp range: [{min(timestamps):.6f}, {max(timestamps):.6f}]")
    else:
        # Default: use first camera's timestamp
        timestamps = [float(train_cameras[0].time)]
        print(f"No timestamp specified, using first camera's time: {timestamps[0]:.6f}")

    # Extract mesh for each timestamp
    print(f"\n{'='*70}")
    print(f"Starting extraction of {len(timestamps)} mesh(es)")
    print(f"{'='*70}")

    extracted_meshes = []
    for i, ts in enumerate(timestamps):
        print(f"\n[{i+1}/{len(timestamps)}] Processing timestamp {ts:.6f}...")
        mesh_path = extract_mesh_at_timestamp(
            args, ts, gaussians, scene, pipe, bg_color, output_dir
        )
        if mesh_path:
            extracted_meshes.append((ts, mesh_path))

    # Summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully extracted {len(extracted_meshes)}/{len(timestamps)} meshes:")
    for ts, path in extracted_meshes:
        print(f"  t={ts:.6f}: {path}")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"{'='*70}")
