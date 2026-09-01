import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

import numpy as np
import torch
from argparse import ArgumentParser
from os import makedirs

import open3d as o3d

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.mesh_utils import GaussianExtractor, post_process_mesh


if __name__ == "__main__":
    parser = ArgumentParser(description="Dynamic multi-resolution TSDF mesh extraction for Deformable 2DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--timestamp", type=float, required=True)
    parser.add_argument("--timestamp_tolerance", type=float, default=1e-6)
    parser.add_argument("--multires_factors", default=[2, 8, 16], nargs="+", type=int,
                        help="depth_trunc = bounding_radius * factor, one mesh per factor")
    parser.add_argument("--interpolate_cameras", action="store_true",
                        help="Add pseudo-views interpolated between training views for TSDF integration.")
    parser.add_argument("--n_neighbors_to_interpolate", type=int, default=2)
    parser.add_argument("--n_interpolated_cameras_for_each_neighbor", type=int, default=10)
    parser.add_argument("--mesh_res", type=int, default=1024,
                        help="voxel_size = depth_trunc / mesh_res")
    parser.add_argument("--num_cluster", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

    # Select the cameras at the requested timestamp (same logic as
    # extract_mesh_adaptive_tsdf.py) and pin their time so the
    # deformation field is evaluated at this timestamp when rendering.
    cams = scene.getTrainCameras()
    matched_cams = [cam for cam in cams if abs(float(cam.time) - args.timestamp) <= args.timestamp_tolerance]
    if matched_cams:
        cams = matched_cams
    else:
        unique_cams, seen = [], set()
        for cam in cams:
            key = (tuple(np.asarray(cam.R).round(6).reshape(-1)), tuple(np.asarray(cam.T).round(6)))
            if key not in seen:
                seen.add(key)
                unique_cams.append(cam)
        cams = unique_cams
    for cam in cams:
        cam.time = args.timestamp
    print(f"[INFO] Dynamic multires TSDF extraction at t={args.timestamp:.6f} using {len(cams)} cameras.")

    cams_are_gs_cameras = False
    if args.interpolate_cameras:
        import math
        from matcha.dm_scene.cameras import get_cameras_interpolated_between_neighbors
        print(f"[INFO] Pseudo-views interpolated between training views will be used for TSDF integration.")
        print(f"          > Interpolating between {args.n_neighbors_to_interpolate} neighbors for each camera.")
        print(f"          > Interpolating {args.n_interpolated_cameras_for_each_neighbor} views for each neighbor.")
        cams = get_cameras_interpolated_between_neighbors(
            cameras=cams,
            n_neighbors_to_interpolate=args.n_neighbors_to_interpolate,
            n_interpolated_cameras_for_each_neighbor=args.n_interpolated_cameras_for_each_neighbor,
        )

        # The interpolated cameras come from matcha and may miss attributes the
        # 2DGS rendering / TSDF code paths expect. Fill in anything absent.
        def _set(cam, name, value):
            try:
                setattr(cam, name, value)
            except (AttributeError, TypeError):
                object.__setattr__(cam, name, value)

        def _projection_matrix(znear, zfar, fovx, fovy):
            tan_half_fov_y = math.tan(fovy / 2)
            tan_half_fov_x = math.tan(fovx / 2)
            top = tan_half_fov_y * znear
            bottom = -top
            right = tan_half_fov_x * znear
            left = -right
            P = torch.zeros(4, 4)
            z_sign = 1.0
            P[0, 0] = 2 * znear / (right - left)
            P[1, 1] = 2 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        for cam in cams:
            _set(cam, "time", args.timestamp)
            for name, value in (("gt_alpha_mask", None), ("znear", 0.01), ("zfar", 100.0)):
                if not hasattr(cam, name):
                    _set(cam, name, value)
            if not hasattr(cam, "world_view_transform"):
                R = np.asarray(cam.R, dtype=np.float64)
                T = np.asarray(cam.T, dtype=np.float64)
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = T
                _set(cam, "world_view_transform", torch.tensor(w2c).float().cuda().T)
            if not hasattr(cam, "camera_center"):
                _set(cam, "camera_center", cam.world_view_transform.inverse().T[:3, 3])
            if not hasattr(cam, "projection_matrix"):
                _set(cam, "projection_matrix",
                     _projection_matrix(cam.znear, cam.zfar, cam.FoVx, cam.FoVy).float().cuda().T)
            if not hasattr(cam, "full_proj_transform"):
                _set(cam, "full_proj_transform",
                     (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0))
        cams_are_gs_cameras = True
        print(f"[INFO] Total cameras for TSDF integration: {len(cams)}")

    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
    # set the active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(cams)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.model_path, "tsdf_meshes_dynamic_multires", f"t_{args.timestamp:.6f}")
    makedirs(output_dir, exist_ok=True)

    # --- Multi-resolution extraction (ported from render_multires.py) ---
    depth_truncs = []
    meshes = []
    for factor in args.multires_factors:
        print(f"\nExtracting mesh with factor {factor}...")
        depth_trunc = gaussExtractor.radius * factor
        voxel_size = depth_trunc / args.mesh_res
        sdf_trunc = 5.0 * voxel_size
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        meshes.append(mesh)
        depth_truncs.append(depth_trunc)
        print(f"Mesh extracted with depth truncation {depth_trunc} and voxel size {voxel_size}.")

    # --- Merge multi-resolution meshes (ported from render_multires.py) ---
    from matcha.dm_scene.cameras import CamerasWrapper, GSCamera
    if cams_are_gs_cameras:
        gs_cameras = cams
    else:
        gs_cameras = []
        for scene_camera in cams:
            gs_cameras.append(GSCamera(
                colmap_id=scene_camera.colmap_id,
                R=scene_camera.R,
                T=scene_camera.T,
                FoVx=scene_camera.FoVx,
                FoVy=scene_camera.FoVy,
                image=scene_camera.original_image,
                gt_alpha_mask=scene_camera.gt_alpha_mask,
                image_name=scene_camera.image_name,
                uid=scene_camera.uid,
                data_device=scene_camera.data_device,
                image_height=scene_camera.image_height,
                image_width=scene_camera.image_width,
            ))
    cameras_wrapper = CamerasWrapper(gs_cameras)

    p3d_meshes = []
    device = "cuda"
    from matcha.dm_scene.meshes import Meshes, TexturesVertex, remove_faces_from_single_mesh
    print("\n===Merging multi-resolution meshes===")
    for i_mesh, (depth_trunc, mesh) in enumerate(zip(depth_truncs, meshes)):
        print(f"Processing mesh with depth truncation {depth_trunc}...")

        verts = torch.from_numpy(np.asarray(mesh.vertices)).float().to(device)
        faces = torch.from_numpy(np.asarray(mesh.triangles)).long().to(device)
        vert_colors = torch.from_numpy(np.asarray(mesh.vertex_colors)).float().to(device)

        p3d_mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex([vert_colors]),
        )
        empty_mesh = False

        # Identify which faces from lower resolutions are necessary to keep
        necessary_faces = torch.zeros(faces.shape[0], dtype=torch.bool, device=device)

        # Removing faces in the field of view of the cameras, but with depth below the truncation threshold
        if i_mesh > 0:
            # Check which vertices are in the field of view...
            projections = cameras_wrapper.project_points(verts.view(1, -1, 3))  # (n_cameras, n_verts, 2)
            height, width = cameras_wrapper.gs_cameras[0].image_height, cameras_wrapper.gs_cameras[0].image_width
            factors = torch.tensor([[[-width / min(height, width), -height / min(height, width)]]], device=projections.device)  # (1, 1, 2)
            projections = projections / factors  # (n_cameras, n_verts, 2)
            visible_mask = (projections[..., 0] > -1.0) & (projections[..., 0] < 1.0) & (projections[..., 1] > -1.0) & (projections[..., 1] < 1.0)  # (n_cameras, n_verts)

            # ... and which are close to the camera
            depths = cameras_wrapper.transform_points_world_to_view(verts.view(1, -1, 3))[..., 2]  # (n_cameras, n_verts)
            close_verts = (depths < depth_truncs[i_mesh - 1])

            non_valid_verts = (visible_mask & close_verts).any(dim=0)  # (n_verts)
            non_valid_faces = non_valid_verts[faces].all(dim=-1)

            # Remove the corresponding faces
            try:
                p3d_mesh = remove_faces_from_single_mesh(p3d_mesh, faces_to_keep_mask=(~non_valid_faces) | necessary_faces)
            except Exception:
                print(f"Error removing faces for mesh {i_mesh}. Empty mesh?")
                empty_mesh = True

        if not empty_mesh:
            p3d_meshes.append(p3d_mesh)

    from pytorch3d.structures import join_meshes_as_scene
    full_mesh = join_meshes_as_scene(p3d_meshes)
    verts = full_mesh.verts_packed()
    faces = full_mesh.faces_packed()
    vert_colors = full_mesh.textures.verts_features_packed()
    # Creates an open3d mesh from the pytorch3d mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors.cpu().numpy())

    mesh_path = os.path.join(output_dir, "multires_tsdf.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print("mesh saved at {}".format(mesh_path))

    # post-process the mesh, keeping the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    post_path = os.path.join(output_dir, "multires_tsdf_post.ply")
    o3d.io.write_triangle_mesh(post_path, mesh_post)
    print("post-processed mesh saved at {}".format(post_path))
