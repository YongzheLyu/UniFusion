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
    parser = ArgumentParser(description="Dynamic classic TSDF mesh extraction for Deformable 2DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--timestamp", type=float, required=True)
    parser.add_argument("--timestamp_tolerance", type=float, default=1e-6)
    parser.add_argument("--depth_trunc_factor", type=float, default=2.0,
                        help="depth_trunc = bounding_radius * factor")
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
    print(f"[INFO] Dynamic TSDF extraction at t={args.timestamp:.6f} using {len(cams)} cameras.")

    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
    # set the active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(cams)

    depth_trunc = gaussExtractor.radius * args.depth_trunc_factor
    voxel_size = depth_trunc / args.mesh_res
    sdf_trunc = 5.0 * voxel_size
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.model_path, "tsdf_meshes_dynamic", f"t_{args.timestamp:.6f}")
    makedirs(output_dir, exist_ok=True)

    mesh_path = os.path.join(output_dir, "tsdf_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print("mesh saved at {}".format(mesh_path))

    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    post_path = os.path.join(output_dir, "tsdf_mesh_post.ply")
    o3d.io.write_triangle_mesh(post_path, mesh_post)
    print("post-processed mesh saved at {}".format(post_path))
