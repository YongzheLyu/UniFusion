import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import gradio
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import torch

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever

import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import cv2
from matcha.dm_utils.dataset_readers import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
# from colmap.read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary
from colmap.read_write_model import Camera, Image, Point3D,  write_cameras_binary, write_images_binary, write_points3D_binary, write_points3D_text, rotmat2qvec
from colmap.read_write_model import write_cameras_text, write_images_text, write_points3D_text
import open3d as o3d

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any


def run_mast3r_sfm(
    scene_path: str,
    output_dir: str,
    weights_path: str,
    retrieval_model: str,
    min_conf_thr: float = 0.,
    matching_conf_thr: float = 0.,
    n_coarse_iterations: int = 1000,
    n_refinement_iterations: int = 1000,
    TSDF_thresh: float = 0.,
    fix_focal: bool = False,
    fix_principal_point: bool = False,
    fix_rotation: bool = False,
    fix_translation: bool = False,
    n_images: int = 10,
    use_all_images: bool = False,
    image_idx: Optional[list] = None,
    randomize_images: bool = False,
    image_size: int = 512,
    max_window_size: int = 20,
    max_refid: int = 10,
    use_calibrated_poses: bool = False,
    save_glb: bool = False,
    gpu: int = 0,
    output_conf_thr: float = .1,
    align_camera_locations: bool = False,
    not_first_frame: bool = False,
    model: Optional[Any] = None,
    device: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run MASt3R SfM on a set of images.

    Args:
        scene_path: Path to the scene data containing images
        output_dir: Path to the output directory
        weights_path: Path to the MASt3R model weights
        retrieval_model: Path to the retrieval model weights
        min_conf_thr: Minimum confidence threshold
        matching_conf_thr: Matching confidence threshold
        n_coarse_iterations: Number of coarse iterations
        n_refinement_iterations: Number of refinement iterations
        TSDF_thresh: TSDF threshold
        fix_focal: Fix camera focal length
        fix_principal_point: Fix camera principal point
        fix_rotation: Fix camera rotation
        fix_translation: Fix camera translation
        n_images: Number of images to use for optimization
        use_all_images: Use all images for optimization
        image_idx: List of view indices to use for optimization
        randomize_images: Shuffle training images before sampling
        image_size: Size of input images
        max_window_size: Maximum window size
        max_refid: Maximum reference image id
        use_calibrated_poses: Use calibrated camera parameters
        save_glb: Save the optimized scene as a .glb file
        gpu: Index of GPU device to use
        output_conf_thr: Confidence threshold for COLMAP format outputs
        align_camera_locations: Align camera locations
        not_first_frame: Use first frame poses for initialization
        model: Pre-loaded MASt3R model (if None, will load one)
        device: Pre-configured device (if None, will create one)
        verbose: Print debug information

    Returns:
        Dictionary containing output information
    """
    import gc
    import os
    import tempfile
    import shutil
    import cv2
    import numpy as np
    import torch
    import math
    import trimesh
    from scipy.spatial.transform import Rotation
    import open3d as o3d

    mast3r.utils.path_to_dust3r  # noqa
    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
    from mast3r.image_pairs import make_pairs
    from mast3r.retrieval.processor import Retriever
    from mast3r.utils.misc import hash_md5
    from dust3r.utils.image import load_images
    from dust3r.utils.device import to_numpy
    from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
    from colmap.read_write_model import Camera, Image, Point3D, write_cameras_binary, write_images_binary, write_points3D_binary, write_points3D_text, write_cameras_text, write_images_text
    from matcha.dm_utils.dataset_readers import qvec2rotmat

    # Set device
    if device is None:
        torch.cuda.set_device(gpu)
        device = torch.device(torch.cuda.current_device())
    else:
        torch.cuda.set_device(device)

    # Load MASt3R model if not provided
    model_loaded = False
    if model is None:
        if verbose:
            print('\\nLoading MASt3R model...')
        torch.serialization.add_safe_globals([argparse.Namespace])
        model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
        torch.serialization.add_safe_globals([argparse.Namespace])
        model_loaded = True
        if verbose:
            print('MASt3R model loaded.')

    def print_debug(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    scene_path = scene_path
    use_calibrated_poses = use_calibrated_poses
    dir_content = os.listdir(scene_path)
    use_subdir = True
    n_all_images = 0
    src_intrinsics=None
    for content in dir_content:
        if (
            content.endswith('.jpg') or content.endswith('.png') or content.endswith('.jpeg')
            or content.endswith('.JPG') or content.endswith('.PNG') or content.endswith('.JPEG')
        ):
            use_subdir = False
            n_all_images += 1
    if not use_subdir:
        if verbose:
            print(f"{n_all_images} images found in {scene_path}.")
    else:
        dir_content = os.listdir(os.path.join(scene_path, "images"))
        for content in dir_content:
            if (
                content.endswith('.jpg') or content.endswith('.png') or content.endswith('.jpeg')
                or content.endswith('.JPG') or content.endswith('.PNG') or content.endswith('.JPEG')
            ):
                n_all_images += 1
        if verbose:
            print(f"{n_all_images} images found in {scene_path}/images/.")
        if use_calibrated_poses:
            src_scale_mats = None
            print("abababa")
            if os.path.exists(f'{scene_path}/sparse/0'):
                src_camera_data = read_intrinsics_binary(f'{scene_path}/sparse/0/cameras.bin')
                src_image_data = read_extrinsics_binary(f'{scene_path}/sparse/0/images.bin')

                src_intrinsics = {}
                src_extrinsics = {}
                for k in src_image_data:
                    img_name = src_image_data[k].name

                    cam = src_camera_data[src_image_data[k].camera_id]
                    if cam.model == 'PINHOLE':
                        fx,fy,cx,cy = cam.params
                        print("115115")
                        print(cam.params)
                    else:
                        raise NotImplementedError('Only PINHOLE model is supported for now.')
                    src_intrinsics[img_name] = np.array([
                        [fx, 0., cx],
                        [0., fy, cy],
                        [0., 0., 1.]
                    ])

                    T = np.eye(4)
                    T[:3,:3] = qvec2rotmat(src_image_data[k].qvec)
                    T[:3,3] = src_image_data[k].tvec
                    src_extrinsics[img_name] = T
            elif os.path.exists(f'{scene_path}/cameras.npz'):
                camera_dict = np.load(f'{scene_path}/cameras.npz')
                scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_all_images)]
                world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_all_images)]
                src_intrinsics = {}
                src_extrinsics = {}
                src_scale_mats = {}
                for k, img_name in enumerate(np.sort(os.listdir(f'{scene_path}/images'))):
                    scale_mat, world_mat = scale_mats[k], world_mats[k]
                    P = world_mat @ scale_mat
                    P = P[:3, :4]

                    K,R,t = cv2.decomposeProjectionMatrix(P)[:3]
                    K = K/K[2,2]
                    cam2w = np.eye(4, dtype=np.float32)
                    cam2w[:3, :3] = R.transpose()
                    cam2w[:3,3] = (t[:3] / t[3])[:,0]

                    src_intrinsics[img_name] = K
                    src_extrinsics[img_name] = np.linalg.inv(cam2w)
                    src_scale_mats[img_name] = scale_mat
            else:
                raise FileNotFoundError(f'Calibration data ({scene_path}/sparse/0/) not found.')

    src_scale_mats = None

    # 当 use_calibrated_poses=True 时，优先读取第一帧的 COLMAP 数据作为 preset pose
    if use_calibrated_poses and not not_first_frame:
        # 尝试从第一帧读取预设的 COLMAP 数据
        base_output = os.path.dirname(output_dir)
        while os.path.basename(base_output):
            if os.path.exists(os.path.join(base_output, 'frame_00000')):
                break
            base_output = os.path.dirname(base_output)
        first_frame_scene = os.path.join(base_output, 'frame_00000/images')
        colmap_dir = os.path.join(base_output, 'frame_00000')
        if verbose:
            print(f"[INFO] Looking for preset pose at: {first_frame_scene}")
        if os.path.exists(f'{colmap_dir}/sparse/0'):
            src_camera_data = read_intrinsics_binary(f'{colmap_dir}/sparse/0/cameras.bin')
            src_image_data = read_extrinsics_binary(f'{colmap_dir}/sparse/0/images.bin')

            src_intrinsics = {}
            src_extrinsics = {}
            for k in src_image_data:
                img_name = src_image_data[k].name

                cam = src_camera_data[src_image_data[k].camera_id]
                if cam.model == 'PINHOLE':
                    fx,fy,cx,cy = cam.params
                else:
                    raise NotImplementedError('Only PINHOLE model is supported for now.')
                src_intrinsics[img_name] = np.array([
                    [fx, 0., cx],
                    [0., fy, cy],
                    [0., 0., 1.]
                ])

                T = np.eye(4)
                T[:3, :3] = qvec2rotmat(src_image_data[k].qvec)
                T[:3, 3] = src_image_data[k].tvec
                src_extrinsics[img_name] = T

            if verbose:
                print(f"[INFO] Loaded preset pose from: {first_frame_scene}/sparse/0")
                print(f"[INFO] Found {len(src_intrinsics)} cameras with preset poses")
        elif verbose:
            print("[WARNING] Preset pose not found, falling back to scene calibration")
    if not_first_frame and src_intrinsics is None:
        if verbose:
            print("[INFO] Using first frame poses.")
            print(f"Looking for first frame scene at base path: {os.path.dirname(os.path.basename(output_dir))}")
        base_output = os.path.dirname(output_dir)
        while os.path.basename(base_output):
            if os.path.exists(os.path.join(base_output, 'frame_00000')):
                break
            base_output = os.path.dirname(base_output)
        first_frame_scene = os.path.join(base_output, 'frame_00000/mast3r_sfm')
        print("first_frame_scene:", first_frame_scene)
        if verbose:
            print(f"First frame scene path: {first_frame_scene}")
        if os.path.exists(f'{first_frame_scene}/sparse/0'):
            src_camera_data = read_intrinsics_binary(f'{first_frame_scene}/sparse/0/cameras.bin')
            src_image_data = read_extrinsics_binary(f'{first_frame_scene}/sparse/0/images.bin')

            src_intrinsics = {}
            src_extrinsics = {}
            for k in src_image_data:
                img_name = src_image_data[k].name

                cam = src_camera_data[src_image_data[k].camera_id]
                if cam.model == 'PINHOLE':
                    fx,fy,cx,cy = cam.params
                    cx = round(cx)
                    cy = round(cy)
                    print("cx:", cx)
                else:
                    raise NotImplementedError('Only PINHOLE model is supported for now.')
                src_intrinsics[img_name] = np.array([
                    [fx, 0., cx],
                    [0., fy, cy],
                    [0., 0., 1.]
                ])

                T = np.eye(4)
                T[:3,:3] = qvec2rotmat(src_image_data[k].qvec)
                T[:3,3] = src_image_data[k].tvec
                src_extrinsics[img_name] = T

    image_dir = os.path.join(scene_path, "images") if use_subdir else scene_path

    # Parameters
    use_all_images = use_all_images
    n_images = n_all_images if use_all_images else n_images
    if image_idx is not None:
        use_all_images = False
        n_images = len(image_idx)
    image_size = image_size
    scenegraph_type = 'retrieval'
    min_conf_thr = min_conf_thr
    matching_conf_thr = matching_conf_thr

    winsize = min(max_window_size, n_images)  # 0 < winsize <= n_images
    refid = min(n_images - 1, max_refid)  # 1 <= refid <= n_images - 1

    shared_intrinsics = True
    cam_size = 0.2
    mask_sky = False
    clean_depth = True

    fix_focal = fix_focal
    fix_pp = fix_principal_point
    fix_rotation = fix_rotation
    fix_translation = fix_translation

    lr1 = 0.07
    niter1 = n_coarse_iterations
    lr2 = 0.01
    niter2 = n_refinement_iterations
    optim_level = 'refine+depth'

    TSDF_thresh = TSDF_thresh

    output_conf_thr = output_conf_thr

    os.makedirs(output_dir, exist_ok=True)
    # Use a unique temporary directory per process to avoid conflicts in multi-threading
    _outdir = os.path.join(output_dir, '.tmp_mast3r')
    os.makedirs(_outdir, exist_ok=True)
    if verbose:
        print_debug("========== PARAMETERS ==========")
        print_debug(f"Scene path: {scene_path}")
        print_debug(f"Image directory: {image_dir}")
        print_debug(f"Output directory: {output_dir}")
        print_debug(f"Number of images: {n_images}")
        print_debug(f"Use all images: {use_all_images}")
        print_debug(f"Image size: {image_size}")
        print_debug(f"Scene graph type: {scenegraph_type}")
        print_debug(f"Minimum confidence threshold: {min_conf_thr}")
        print_debug(f"Matching confidence threshold: {matching_conf_thr}")
        print_debug(f"Window size: {winsize}")
        print_debug(f"Reference image id: {refid}")
        print_debug(f"Shared intrinsics: {shared_intrinsics}")
        print_debug(f"Camera size: {cam_size}")
        print_debug(f"Mask sky: {mask_sky}")
        print_debug(f"Clean depth: {clean_depth}")
        print_debug(f"Learning rate 1: {lr1}")
        print_debug(f"Number of coarse iterations: {niter1}")
        print_debug(f"Learning rate 2: {lr2}")
        print_debug(f"Number of refinement iterations: {niter2}")
        print_debug(f"Optimization level: {optim_level}")
        print_debug(f"TSDF threshold: {TSDF_thresh}")
        print_debug(f"Output confidence threshold: {output_conf_thr}")
        print_debug("================================\\n")

    # Helper functions
    class SparseGAState:
        def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
            self.sparse_ga = sparse_ga
            self.cache_dir = cache_dir
            self.outfile_name = outfile_name
            self.should_delete = should_delete

        def __del__(self):
            if not self.should_delete:
                return
            if self.cache_dir is not None and os.path.isdir(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            self.cache_dir = None
            if self.outfile_name is not None and os.path.isfile(self.outfile_name):
                os.remove(self.outfile_name)
            self.outfile_name = None

    def _build_first_frame_mapping(src_intrinsics_dict):
        """Build mapping from camera prefix to first frame filename."""
        mapping = {}
        for name in src_intrinsics_dict.keys():
            stem = Path(name).stem
            parts = stem.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                prefix = '_'.join(parts[:-1])
                frame_id = int(parts[-1])
            else:
                continue

            if prefix not in mapping:
                mapping[prefix] = (frame_id, name)
            else:
                prev_frame, prev_name = mapping[prefix]
                if frame_id < prev_frame:
                    mapping[prefix] = (frame_id, name)
        if verbose:
            print(f"First frame mapping: {mapping}")
        return {k: v[1] for k, v in mapping.items()}

    def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                    cam_color=None, as_pointcloud=False,
                                    transparent_cams=False, silent=False):
        assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
        pts3d = to_numpy(pts3d)
        imgs = to_numpy(imgs)
        focals = to_numpy(focals)
        cams2world = to_numpy(cams2world)

        scene = trimesh.Scene()

        if as_pointcloud:
            pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
            col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
            valid_msk = np.isfinite(pts.sum(axis=1))
            pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
            scene.add_geometry(pct)
        else:
            meshes = []
            for i in range(len(imgs)):
                pts3d_i = pts3d[i].reshape(imgs[i].shape)
                msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
                meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
            mesh = trimesh.Trimesh(**cat_meshes(meshes))
            scene.add_geometry(mesh)

        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(scene, pose_c2w, camera_edge_color,
                        None if transparent_cams else imgs[i], focals[i],
                        imsize=imgs[i].shape[1::-1], screen_width=cam_size)

        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
        if not silent:
            print('(exporting 3D scene to', outfile, ')')
        scene.export(file_obj=outfile)
        return outfile

    # Load images
    if verbose:
        print('\\nLoading images...')
    all_image_names = np.sort(os.listdir(image_dir))
    
    print(all_image_names)
    if image_idx is not None:
        image_names = [all_image_names[i] for i in image_idx]
    elif use_all_images:
        image_names = all_image_names.tolist()
    else:
        image_names = [all_image_names[i * (len(all_image_names) // (n_images - 1))] for i in range(n_images)]

    filelist = [os.path.join(image_dir, image_name) for image_name in image_names]
    if verbose:
        print(f'filelist: {filelist}')
    silent = True
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    print(f'Loaded {len(imgs)} images.')
    print(f'Images have shape: {imgs[0]["img"].shape}')

    if not use_calibrated_poses:
        os.makedirs(f'{output_dir}/images', exist_ok=True)
        for idx_img, img_path in enumerate(filelist):
            img_fname = img_path.split('/')[-1]
            src_img = cv2.imread(img_path, 1)

            if verbose:
                print_debug(f"\\nSource image has shape: {src_img.shape}")
            if True:
                _height_resized, _width_resized = imgs[idx_img]['img'].size()[-2:]
                _height_original, _width_original = src_img.shape[:2]

                _width_crop = int(round(max(_height_original, _width_original) / image_size * _width_resized))
                _height_crop = int(round(max(_height_original, _width_original) / image_size * _height_resized))

                if verbose:
                    print_debug(f"Width crop: {_width_crop}")
                    print_debug(f"Height crop: {_height_crop}")

                if verbose:
                    print_debug(f"Source image shape (before cropping): {src_img.shape}")

                src_img = src_img[_height_original//2 - _height_crop//2:_height_original//2 + _height_crop//2,
                                   _width_original//2 - _width_crop//2:_width_original//2 + _width_crop//2]
                if verbose:
                    print_debug(f"Source image shape (after cropping): {src_img.shape}")
                cv2.imwrite(f'{output_dir}/images/{img_fname}', src_img)
                filelist[idx_img] = f'{output_dir}/images/{img_fname}'

        imgs = load_images(filelist, size=image_size, verbose=not silent)
        if verbose:
            print(f'Images have shape: {imgs[0]["img"].shape}')

    intrinsics = None
    extrinsics = None

    if not_first_frame:
        print_debug("[INFO] Using first frame poses.")
        first_frame_map = _build_first_frame_mapping(src_intrinsics)

        first_frame_image_names = []
        for img_name in image_names:
            stem = Path(img_name).stem
            parts = stem.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                cam_prefix = '_'.join(parts[:-1])
                if cam_prefix in first_frame_map:
                    ref_name = first_frame_map[cam_prefix]
                else:
                    ref_name = next(iter(src_intrinsics.keys()))
                first_frame_image_names.append(ref_name)
            else:
                ref_name = next(iter(src_intrinsics.keys()))
                first_frame_image_names.append(ref_name)

        intrinsics = np.stack([src_intrinsics[n] for n in first_frame_image_names], axis=0)
        extrinsics = np.stack([src_extrinsics[n] for n in first_frame_image_names], axis=0)
        calibrated_img_sizes = []

        if verbose:
            print_debug('original intrinsics')
            print_debug(intrinsics)
            print_debug('original extrinsics')
            print_debug(extrinsics)

        os.makedirs(f'{output_dir}/images', exist_ok=True)
        for idx_img, img_path in enumerate(filelist):
            img_fname = img_path.split('/')[-1]
            src_img = cv2.imread(img_path, 1)
            src_K = intrinsics[idx_img]

            src_height, src_width = src_img.shape[:2]
            src_dist = np.array([0., 0., 0., 0.])
            src_pp = src_K[:2,2]

            tar_pp = np.array([
                int(min(src_pp[0], src_width - src_pp[0])),
                int(min(src_pp[1], src_height - src_pp[1])),
            ])

            if verbose:
                print_debug(f"Tar pp: {tar_pp}")
            centered_height = 2 * tar_pp[1]
            centered_width = 2 * tar_pp[0]
            if verbose:
                print_debug(f"Height centered: {centered_height}")
                print_debug(f"Width centered: {centered_width}")
            height_resized = imgs[idx_img]['img'].shape[-2]
            width_resized = imgs[idx_img]['img'].shape[-1]
            if verbose:
                print_debug(f"Height resized: {height_resized}")
                print_debug(f"Width resized: {width_resized}")
            width_crop = int(round(max(centered_height, centered_width) / image_size * width_resized))
            height_crop = int(round(max(centered_height, centered_width) / image_size * height_resized))
            tar_pp = np.array([width_crop//2, height_crop//2])
            if verbose:
                print_debug(f"New tar pp: {tar_pp}")

            w_, h_ = tar_pp * 2
            s_w = int(round(w_*image_size/max(w_, h_))) / w_
            s_h = int(round(h_*image_size/max(w_, h_))) / h_
            tar_focal = 0.5 * (src_K[0,0] + src_K[1,1])
            tar_focal_x = tar_focal * np.sqrt(s_w * s_h) / s_w
            tar_focal_y = tar_focal * np.sqrt(s_w * s_h) / s_h
            tar_K = src_K.copy()
            tar_K[:2,2] = tar_pp
            tar_K[0,0] = tar_focal_x
            tar_K[1,1] = tar_focal_y
            map1, map2 = cv2.initUndistortRectifyMap(
                src_K, src_dist, None,
                tar_K, 2 * tar_pp,
                cv2.CV_32FC2
            )
            new_img = cv2.remap(
                src_img,
                map1, map2,
                interpolation=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            calibrated_img_sizes.append((new_img.shape[0], new_img.shape[1]))

            if verbose:
                print_debug(f"\\nSource image has shape: {src_img.shape}")
                print_debug(f"Calibrated image has shape: {new_img.shape}")

            if verbose:
                print_debug(f"Source image has shape: {src_img.shape}")
                print_debug(f"Final image has shape: {new_img.shape}")

            cv2.imwrite(f'{output_dir}/images/{img_fname}', new_img)
            filelist[idx_img] = f'{output_dir}/images/{img_fname}'
            intrinsics[idx_img] = tar_K

        imgs = load_images(filelist, size=image_size, verbose=not silent)
        if verbose:
            print_debug(f'Images have shape: {imgs[0]["img"].shape}')

        if verbose:
            print_debug('intrinsics after cropping')
            print_debug(intrinsics)

        for idx_img, img in enumerate(imgs):
            height_resized, width_resized = img['img'].size()[-2:]

            new_img = cv2.imread(filelist[idx_img], 1)
            height_original, width_original = new_img.shape[:2]

            width_resized_simple = int(round(width_original*image_size/max(width_original, height_original)))
            height_resized_simple = int(round(height_original*image_size/max(width_original, height_original)))
            ofs_u = width_resized_simple//2 - width_resized//2
            ofs_v = height_resized_simple//2 - height_resized//2

            intrinsics[idx_img][0,:] *= (width_resized_simple / width_original)
            intrinsics[idx_img][1,:] *= (height_resized_simple / height_original)
            intrinsics[idx_img][0,2] -= ofs_u
            intrinsics[idx_img][1,2] -= ofs_v

        imgs = load_images(filelist, size=image_size, verbose=not silent)
        if verbose:
            print_debug(f'Images have shape: {imgs[0]["img"].shape}')

        if verbose:
            print_debug('intrinsics after resizing')
            print_debug(intrinsics)

    if use_calibrated_poses and not not_first_frame:
        print_debug("[INFO] Using calibrated poses.")
        for img_name in image_names:
            print(img_name)
        intrinsics = np.stack([src_intrinsics[img_name] for img_name in image_names], axis=0)
        extrinsics = np.stack([src_extrinsics[img_name] for img_name in image_names], axis=0)
        calibrated_img_sizes = []

        if verbose:
            print_debug('original intrinsics')
            print_debug(intrinsics)
            print_debug('original extrinsics')
            print_debug(extrinsics)

        os.makedirs(f'{output_dir}/images', exist_ok=True)
        for idx_img, img_path in enumerate(filelist):
            img_fname = img_path.split('/')[-1]
            src_img = cv2.imread(img_path, 1)
            src_K = intrinsics[idx_img]

            src_height, src_width = src_img.shape[:2]
            src_dist = np.array([0., 0., 0., 0.])
            src_pp = src_K[:2,2]

            tar_pp = np.array([
                int(min(src_pp[0], src_width - src_pp[0])),
                int(min(src_pp[1], src_height - src_pp[1])),
            ])

            if verbose:
                print_debug(f"Tar pp: {tar_pp}")
            centered_height = 2 * tar_pp[1]
            centered_width = 2 * tar_pp[0]
            if verbose:
                print_debug(f"Height centered: {centered_height}")
                print_debug(f"Width centered: {centered_width}")
            height_resized = imgs[idx_img]['img'].shape[-2]
            width_resized = imgs[idx_img]['img'].shape[-1]
            if verbose:
                print_debug(f"Height resized: {height_resized}")
                print_debug(f"Width resized: {width_resized}")
            width_crop = int(round(max(centered_height, centered_width) / image_size * width_resized))
            height_crop = int(round(max(centered_height, centered_width) / image_size * height_resized))
            tar_pp = np.array([width_crop//2, height_crop//2])
            if verbose:
                print_debug(f"New tar pp: {tar_pp}")

            w_, h_ = tar_pp * 2
            s_w = int(round(w_*image_size/max(w_, h_))) / w_
            s_h = int(round(h_*image_size/max(w_, h_))) / h_
            tar_focal = 0.5 * (src_K[0,0] + src_K[1,1])
            tar_focal_x = tar_focal * np.sqrt(s_w * s_h) / s_w
            tar_focal_y = tar_focal * np.sqrt(s_w * s_h) / s_h
            tar_K = src_K.copy()
            tar_K[:2,2] = tar_pp
            tar_K[0,0] = tar_focal_x
            tar_K[1,1] = tar_focal_y
            map1, map2 = cv2.initUndistortRectifyMap(
                src_K, src_dist, None,
                tar_K, 2 * tar_pp,
                cv2.CV_32FC2
            )
            new_img = cv2.remap(
                src_img,
                map1, map2,
                interpolation=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            calibrated_img_sizes.append((new_img.shape[0], new_img.shape[1]))

            if verbose:
                print_debug(f"\\nSource image has shape: {src_img.shape}")
                print_debug(f"Calibrated image has shape: {new_img.shape}")

            if verbose:
                print_debug(f"Source image has shape: {src_img.shape}")
                print_debug(f"Final image has shape: {new_img.shape}")
            cv2.imwrite(f'{output_dir}/images/{img_fname}', new_img)

            intrinsics[idx_img] = tar_K
            filelist[idx_img] = f'{output_dir}/images/{img_fname}'
        imgs = load_images(filelist, size=image_size, verbose=not silent)
        if verbose:
            print_debug(f'Images have shape: {imgs[0]["img"].shape}')

        if verbose:
            print_debug('intrinsics after cropping')
            print_debug(intrinsics)

        for idx_img, img in enumerate(imgs):
            height_resized, width_resized = img['img'].size()[-2:]

            new_img = cv2.imread(filelist[idx_img], 1)
            height_original, width_original = new_img.shape[:2]

            width_resized_simple = int(round(width_original*image_size/max(width_original, height_original)))
            height_resized_simple = int(round(height_original*image_size/max(width_original, height_original)))
            ofs_u = width_resized_simple//2 - width_resized//2
            ofs_v = height_resized_simple//2 - height_resized//2

            intrinsics[idx_img][0,:] *= (width_resized_simple / width_original)
            intrinsics[idx_img][1,:] *= (height_resized_simple / height_original)
            intrinsics[idx_img][0,2] -= ofs_u
            intrinsics[idx_img][1,2] -= ofs_v

        imgs = load_images(filelist, size=image_size, verbose=not silent)
        if verbose:
            print_debug(f'Images have shape: {imgs[0]["img"].shape}')

        if verbose:
            print_debug('intrinsics after resizing')
            print_debug(intrinsics)

    # Load retrieval model
    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))
        scene_graph_params.append(str(refid))
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)
        del retriever
        torch.cuda.empty_cache()

    # Build scene graph
    if verbose:
        print('\\nStart building scene graph...')
    t0 = time.time()

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    if optim_level == 'coarse':
        niter2 = 0

    cache_dir = os.path.join(_outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    if verbose:
        print('Done building scene graph.')

    # Run sparse global alignment
    if verbose:
        print('Start global alignment...')
    kw = {
        'init': {},
        'opt_pp': True if not fix_pp else False,
        'opt_focal': True if not fix_focal else False,
        'opt_quat': True if not fix_rotation else False,
        'opt_tran': True,
        'opt_size': True,
    }
    if use_calibrated_poses:
        for idx_img, img_path in enumerate(filelist):
            kw['init'][img_path] = {
                'intrinsics': torch.from_numpy(intrinsics[idx_img].astype(np.float32)).to(device),
                'cam2w': torch.inverse(torch.from_numpy(extrinsics[idx_img].astype(np.float32)).to(device)),
            }

    if not_first_frame:
        kw = {
            'init': {},
            'opt_pp': False,
            'opt_focal': False,
            'opt_quat': False,
            'opt_tran': True,
            'opt_size': True,
        }
        for idx_img, img_path in enumerate(filelist):
            kw['init'][img_path] = {
                'intrinsics': torch.from_numpy(intrinsics[idx_img].astype(np.float32)).to(device),
                'cam2w': torch.inverse(torch.from_numpy(extrinsics[idx_img].astype(np.float32)).to(device)),
            }

    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)

    outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=_outdir)
    scene_state = SparseGAState(scene, False, cache_dir, outfile_name)
    if verbose:
        print('Done global alignment.')
        print(f'Optimization complete in {time.time() - t0:.2f}s.')

    # Get optimized values from scene
    if verbose:
        print('Saving optimized scene...')
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    pps = scene.get_principal_points().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        if verbose:
            print('TSDF post-processing...')
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    if not_first_frame:
        if verbose:
            print('[INFO] Aligning camera locations...')
        calib_cam_locations = []
        est_cam_locations = []
        for idx_view, img_name in enumerate(image_names):
            first_frame_map = _build_first_frame_mapping(src_intrinsics)
            stem = Path(img_name).stem
            parts = stem.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                cam_prefix = '_'.join(parts[:-1])
                if cam_prefix in first_frame_map:
                    ref_name = first_frame_map[cam_prefix]
                else:
                    ref_name = next(iter(src_extrinsics.keys()))
            else:
                ref_name = next(iter(src_extrinsics.keys()))

            calib_cam_locations.append(np.linalg.inv(src_extrinsics[ref_name])[:3,3])
            est_cam_locations.append(cams2world[idx_view][:3,3].cpu().numpy())
        calib_cam_locations = np.stack(calib_cam_locations, axis=0)
        est_cam_locations = np.stack(est_cam_locations, axis=0)

        if True:
            x = est_cam_locations - np.mean(est_cam_locations, axis=0, keepdims=True)
            y = calib_cam_locations - np.mean(calib_cam_locations, axis=0, keepdims=True)
            scale_est2calib = np.sum(x * y) / np.sum(x**2)
            ofs_est2calib = np.mean(calib_cam_locations, axis=0) - scale_est2calib * np.mean(est_cam_locations, axis=0)

            mean_residual = np.mean(np.linalg.norm(scale_est2calib * est_cam_locations + ofs_est2calib - calib_cam_locations, axis=-1))
            if verbose:
                print_debug('Scale and translation offsets:', scale_est2calib, ofs_est2calib)
                print_debug('Mean residual of camera locations:', mean_residual)

            cams2world[:,:3,3] = scale_est2calib * cams2world[:,:3,3] + torch.from_numpy(ofs_est2calib).to(cams2world.device)
            if verbose:
                print_debug('modified cams2world', torch.inverse(cams2world).numpy())
            for i_ in range(len(pts3d)):
                pts3d[i_] = scale_est2calib * pts3d[i_] + ofs_est2calib

        cams2world = torch.from_numpy(np.linalg.inv(extrinsics))

        if verbose:
            print_debug('output extrinsics:')
            print_debug(torch.inverse(cams2world).numpy())

        if src_scale_mats is not None:
            for i_ in range(len(pts3d)):
                S = src_scale_mats[image_names[i_]]
                cam_location = cams2world[i_:i_+1,:3,3].cpu().numpy()
                new_cam_location = (S[:3,:3] @ cam_location.T + S[:3,3:4]).T
                cams2world[i_:i_+1,:3,3] = torch.from_numpy(new_cam_location).to(cams2world.device)
                pts3d[i_] = (S[:3,:3] @ pts3d[i_].T + S[:3,3:4]).T

    # Align the recovered camera locations with calibrated ones
    if use_calibrated_poses and align_camera_locations and not not_first_frame:
        if verbose:
            print('[INFO] Aligning camera locations...')
        calib_cam_locations = []
        est_cam_locations = []
        for idx_view, img_name in enumerate(image_names):
            calib_cam_locations.append(np.linalg.inv(src_extrinsics[img_name])[:3,3])
            est_cam_locations.append(cams2world[idx_view][:3,3].cpu().numpy())
        calib_cam_locations = np.stack(calib_cam_locations, axis=0)
        est_cam_locations = np.stack(est_cam_locations, axis=0)

        if True:
            x = est_cam_locations - np.mean(est_cam_locations, axis=0, keepdims=True)
            y = calib_cam_locations - np.mean(calib_cam_locations, axis=0, keepdims=True)
            scale_est2calib = np.sum(x * y) / np.sum(x**2)
            ofs_est2calib = np.mean(calib_cam_locations, axis=0) - scale_est2calib * np.mean(est_cam_locations, axis=0)

            mean_residual = np.mean(np.linalg.norm(scale_est2calib * est_cam_locations + ofs_est2calib - calib_cam_locations, axis=-1))
            if verbose:
                print_debug('Scale and translation offsets:', scale_est2calib, ofs_est2calib)
                print_debug('Mean residual of camera locations:', mean_residual)

            cams2world[:,:3,3] = scale_est2calib * cams2world[:,:3,3] + torch.from_numpy(ofs_est2calib).to(cams2world.device)
            for i_ in range(len(pts3d)):
                pts3d[i_] = scale_est2calib * pts3d[i_] + ofs_est2calib

        if fix_rotation and fix_translation:
            cams2world = torch.from_numpy(np.linalg.inv(extrinsics))
        elif fix_translation:
            cams2world[:,:3,3] = torch.from_numpy(np.linalg.inv(extrinsics))[:,:3,3]
        elif fix_rotation:
            cams2world[:,:3,:3] = torch.from_numpy(np.linalg.inv(extrinsics))[:,:3,:3]

        if verbose:
            print_debug('output extrinsics:')
            print_debug(torch.inverse(cams2world).numpy())

        if src_scale_mats is not None:
            for i_ in range(len(pts3d)):
                S = src_scale_mats[image_names[i_]]
                cam_location = cams2world[i_:i_+1,:3,3].cpu().numpy()
                new_cam_location = (S[:3,:3] @ cam_location.T + S[:3,3:4]).T
                cams2world[i_:i_+1,:3,3] = torch.from_numpy(new_cam_location).to(cams2world.device)
                pts3d[i_] = (S[:3,:3] @ pts3d[i_].T + S[:3,3:4]).T

    # Save cameras
    output_cameras = {
        'filepaths': [img['instance'] for img in imgs],
        'focals': focals.numpy().tolist(),
        'cams2world': cams2world.numpy().tolist(),
    }
    with open(os.path.join(output_dir, 'cameras.json'), 'w') as f:
        json.dump(output_cameras, f)

    cameras_colmap = {}
    images_colmap = {}
    points3d_colmap = {}
    points3d_id_ofs = 1
    all_xyzs = []
    all_rgbs = []
    for idx_img in range(len(imgs)):
        camera_id = idx_img + 1

        height_original, width_original = cv2.imread(imgs[idx_img]['instance'],0).shape[:2]
        width_resized = int(imgs[idx_img]['img'].size(-1))
        height_resized = int(imgs[idx_img]['img'].size(-2))

        width_resized_simple = int(round(width_original*image_size/max(width_original, height_original)))
        height_resized_simple = int(round(height_original*image_size/max(width_original, height_original)))
        ofs_u = width_resized_simple//2 - width_resized//2
        ofs_v = height_resized_simple//2 - height_resized//2

        scale_w = width_original / width_resized_simple
        scale_h = height_original / height_resized_simple
        cameras_colmap[camera_id] = Camera(
            int(camera_id),
            'PINHOLE',
            width=width_original,
            height=height_original,
            params=np.array([
                focals[idx_img].item() * scale_w,
                focals[idx_img].item() * scale_h,
                (pps[idx_img][0].item() + ofs_u) * scale_w,
                (pps[idx_img][1].item() + ofs_v) * scale_h,
            ])
        )

        w2c = np.linalg.inv(cams2world[idx_img].numpy())
        pixels = np.mgrid[:width_resized, :height_resized].T.reshape(-1, 2)
        pixels_original = pixels * np.array([scale_w, scale_h])
        point3d_ids = []
        xys = []
        img_original = cv2.imread(imgs[idx_img]['instance'],1)[...,::-1]

        for idx_pt2d, (xyz, pt2d) in enumerate(zip(pts3d[idx_img], pixels_original)):
            pixel_conf = confs[idx_img][pixels[idx_pt2d][1], pixels[idx_pt2d][0]]
            if pixel_conf < output_conf_thr:
                continue

            point3d_id = points3d_id_ofs + idx_pt2d
            xys.append(pt2d)
            point3d_ids.append(point3d_id)

            rgb = img_original[int(np.clip(pt2d[1],0,height_original)), int(np.clip(pt2d[0],0,width_original))]
            error = np.array(2., dtype=np.float64)

            image_ids = np.array([camera_id,], dtype=np.int64)
            point2D_idxs=np.array([idx_pt2d,], dtype=np.int64)

            points3d_colmap[point3d_id] = Point3D(
                id=point3d_id,
                xyz=xyz.astype(np.float64),
                rgb=rgb.astype(np.int64),
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
            all_xyzs.append(xyz.astype(np.float64))
            all_rgbs.append(rgb.astype(np.float64) / 255.)
        points3d_id_ofs += len(pts3d[idx_img])

        images_colmap[camera_id] = Image(
            camera_id,
            rotmat2qvec(w2c[:3,:3]).astype(np.float64),
            w2c[:3,3].astype(np.float64),
            camera_id,
            imgs[idx_img]['instance'].split('/')[-1],
            np.array(xys, dtype=np.float64),
            np.array(point3d_ids, dtype=np.int64)
        )

    os.makedirs(f'{output_dir}/sparse/0', exist_ok=True)
    write_cameras_binary(cameras_colmap, f'{output_dir}/sparse/0/cameras.bin')
    write_images_binary(images_colmap, f'{output_dir}/sparse/0/images.bin')
    write_points3D_binary(points3d_colmap, f'{output_dir}/sparse/0/points3D.bin')
    if True:
        write_cameras_text(cameras_colmap, f'{output_dir}/sparse/0/cameras.txt')
        write_images_text(images_colmap, f'{output_dir}/sparse/0/images.txt')
        write_points3D_text(points3d_colmap, f'{output_dir}/sparse/0/points3D.txt')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(all_xyzs, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(all_rgbs, axis=0))
    o3d.io.write_point_cloud(f"{output_dir}/points.ply", pcd)

    # Save pointmaps
    pointmaps_dir = os.path.join(output_dir, 'pointmaps')
    os.makedirs(pointmaps_dir, exist_ok=True)
    for i_image in range(len(imgs)):
        img_path = imgs[i_image]['instance']
        img_name = os.path.basename(img_path).split('.')[0]
        output_pointmap = {
            'rgb': None if use_all_images else rgbimg[i_image].tolist(),
            'points': pts3d[i_image].tolist(),
            'confs': confs[i_image].tolist(),
        }
        with open(os.path.join(pointmaps_dir, f'{img_name}.json'), 'w') as f:
            json.dump(output_pointmap, f)

    if save_glb:
        as_pointcloud = True
        transparent_cams = False
        msk = to_numpy([c > min_conf_thr for c in confs])
        _convert_scene_output_to_glb(
            os.path.join(output_dir, 'scene.glb'),
            rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
            transparent_cams=transparent_cams, cam_size=cam_size, silent=silent
        )
    if verbose:
        print('Scene saved.')

    # Remove temporary files
    if verbose:
        print('\\nCleaning up temporary files...')
    shutil.rmtree(_outdir)
    if verbose:
        print('Temporary files cleaned up.')
        print('\\nInitial SfM complete.')

    # Clean up model if we loaded it
    if model_loaded:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return {
        'output_dir': output_dir,
        'n_images': len(imgs),
        'cams2world': cams2world.numpy(),
    }


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to run MASt3R-SfM on a set of images.')
    
    # Input/Output
    parser.add_argument('-s', '--scene_path', type=str, 
                        help='Path to the scene data to use. Can be a directory containing images, or a colmap output directory.')
    parser.add_argument('-o', '--output_dir', type=str, default=None, 
                        help='Path to the output directory.')
    
    # Mast3r parameters
    parser.add_argument('--weights_path', type=str, 
                        default='./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', 
                        help='Path to the MASt3R model weights.')
    parser.add_argument('--retrieval_model', type=str, 
                        default='./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth', 
                        help='Path to the retrieval model weights.')
    
    # Optimization parameters
    parser.add_argument('--min_conf_thr', type=float, default=0., help='Minimum confidence threshold.')
    parser.add_argument('--matching_conf_thr', type=float, default=0., help='Matching confidence threshold.')
    parser.add_argument('--n_coarse_iterations', type=int, default=1000, help='Number of coarse iterations.')
    parser.add_argument('--n_refinement_iterations', type=int, default=1000, help='Number of refinement iterations.')
    parser.add_argument('--TSDF_thresh', type=float, default=0., help='TSDF threshold.')
    parser.add_argument('--fix_focal', action='store_true', help='Fix camera focal length.')
    parser.add_argument('--fix_principal_point', action='store_true', help='Fix camera principal point.')
    parser.add_argument('--fix_rotation', action='store_true', help='Fix camera rotation.')
    parser.add_argument('--fix_translation', action='store_true', help='Fix camera translation.')
    
    # Data parameters
    parser.add_argument('--n_images', type=int, default=10, help='Number of images to use for optimization, sampled with constant spacing.')
    parser.add_argument('--use_all_images', action='store_true', help='Use all images for optimization.')
    parser.add_argument('--image_idx', type=int, nargs='*', default=None, 
        help='View indices to use for optimization (zero-based indexing). If provided, this will override the --n_images and --use_all_images arguments.')
    parser.add_argument('--randomize_images', action='store_true', help='Shuffle training images before sampling with constant spacing.')
    parser.add_argument('--image_size', type=int, default=512, help='Size of input images.')
    parser.add_argument('--max_window_size', type=int, default=20, help='Maximum window size.')
    parser.add_argument('--max_refid', type=int, default=10, help='Maximum reference image id.')
    parser.add_argument('--use_calibrated_poses', action='store_true', help='Use calibrated camera paramters in COLMAP format.')
    
    # Misc
    parser.add_argument('--save_glb', action='store_true', help='Save the optimized scene as a .glb file.')
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')

    # Post processing    
    parser.add_argument('--output_conf_thr', type=float, default=.1, help='Confidence threshold for COLMAP format outputs.')
    parser.add_argument('--align_camera_locations', action='store_true', help='Align camera locations.')
    parser.add_argument('--not_first_frame', type=str, default='True')

    args = parser.parse_args()
    debug_verbose = True
    def print_debug(*args, **kwargs):
        if debug_verbose:
            print(*args, **kwargs)
    
    # If the user provides a directory containing images, use them.
    # If not, we assume this is a COLMAP dataset, 
    # and images should be located in the scene_path/images subdirectory.
    scene_path = args.scene_path
    use_calibrated_poses = args.use_calibrated_poses
    dir_content = os.listdir(scene_path)
    use_subdir = True
    n_all_images = 0
    
    if args.not_first_frame == 'True':
        not_first_frame = True
    else:
        not_first_frame = False
    
    print("="*50)
    print('not_first_frame:', not_first_frame)
    print('='*50)
    for content in dir_content:
        if (
            content.endswith('.jpg') or content.endswith('.png') or content.endswith('.jpeg')
            or content.endswith('.JPG') or content.endswith('.PNG') or content.endswith('.JPEG')
        ):
            use_subdir = False
            n_all_images += 1
    if not use_subdir:
        print(f"{n_all_images} images found in {scene_path}.")
    else:
        dir_content = os.listdir(os.path.join(scene_path, "images"))
        for content in dir_content:
            if (
                content.endswith('.jpg') or content.endswith('.png') or content.endswith('.jpeg')
                or content.endswith('.JPG') or content.endswith('.PNG') or content.endswith('.JPEG')
            ):
                n_all_images += 1
        print(f"{n_all_images} images found in {scene_path}/images/.")
        if use_calibrated_poses:
            src_scale_mats = None
            
            if os.path.exists(f'{scene_path}/sparse/0'):
                src_camera_data = read_intrinsics_binary(f'{scene_path}/sparse/0/cameras.bin')
                src_image_data = read_extrinsics_binary(f'{scene_path}/sparse/0/images.bin')

                src_intrinsics = {}
                src_extrinsics = {}
                for k in src_image_data:
                    img_name = src_image_data[k].name

                    cam = src_camera_data[src_image_data[k].camera_id]
                    if cam.model == 'PINHOLE':
                        fx,fy,cx,cy = cam.params
                    else:
                        raise NotImplementedError('Only PINHOLE model is supported for now.')
                    src_intrinsics[img_name] = np.array([
                        [fx, 0., cx],
                        [0., fy, cy],
                        [0., 0., 1.]
                    ])

                    T = np.eye(4)
                    T[:3,:3] = qvec2rotmat(src_image_data[k].qvec)
                    T[:3,3] = src_image_data[k].tvec
                    src_extrinsics[img_name] = T
            elif os.path.exists(f'{scene_path}/cameras.npz'):
                camera_dict = np.load(f'{scene_path}/cameras.npz')
                scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_all_images)]
                world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_all_images)]
                src_intrinsics = {}
                src_extrinsics = {}
                src_scale_mats = {}
                for k, img_name in enumerate(np.sort(os.listdir(f'{scene_path}/images'))):
                    scale_mat, world_mat = scale_mats[k], world_mats[k]
                    P = world_mat @ scale_mat
                    P = P[:3, :4]

                    K,R,t = cv2.decomposeProjectionMatrix(P)[:3]
                    K = K/K[2,2]
                    cam2w = np.eye(4, dtype=np.float32)
                    cam2w[:3, :3] = R.transpose()
                    cam2w[:3,3] = (t[:3] / t[3])[:,0]

                    src_intrinsics[img_name] = K
                    src_extrinsics[img_name] = np.linalg.inv(cam2w)
                    src_scale_mats[img_name] = scale_mat
            else:
                raise FileNotFoundError(f'Calibration data ({scene_path}/sparse/0/) not found.')
    if not_first_frame:
            print(args.output_dir)
            base_output = os.path.dirname(os.path.dirname(args.output_dir))
            first_frame_scene = os.path.join(base_output,'frame_00000/mast3r_sfm')
            # print('='*50)
            # print(scene_path)
            # print('='*50)
            # first_frame_scene = scene_path[:-8] + '0' + scene_path[-7:]
            #first_frame_scene[-8] = '0'
            print(first_frame_scene)
            src_scale_mats = None
            if os.path.exists(f'{first_frame_scene}/sparse/0'):
                src_camera_data = read_intrinsics_binary(f'{first_frame_scene}/sparse/0/cameras.bin')
                src_image_data = read_extrinsics_binary(f'{first_frame_scene}/sparse/0/images.bin')

                src_intrinsics = {}
                src_extrinsics = {}
                for k in src_image_data:
                    img_name = src_image_data[k].name

                    cam = src_camera_data[src_image_data[k].camera_id]
                    if cam.model == 'PINHOLE':
                        fx,fy,cx,cy = cam.params
                    else:
                        raise NotImplementedError('Only PINHOLE model is supported for now.')
                    src_intrinsics[img_name] = np.array([
                        [fx, 0., cx],
                        [0., fy, cy],
                        [0., 0., 1.]
                    ])

                    T = np.eye(4)
                    T[:3,:3] = qvec2rotmat(src_image_data[k].qvec)
                    T[:3,3] = src_image_data[k].tvec
                    src_extrinsics[img_name] = T

    image_dir = os.path.join(scene_path, "images") if use_subdir else scene_path
    
    # Parameters
    use_all_images = args.use_all_images
    n_images = n_all_images if use_all_images else args.n_images
    if args.image_idx is not None:
        use_all_images = False
        n_images = len(args.image_idx)
    image_size = args.image_size
    scenegraph_type = 'retrieval'
    min_conf_thr = args.min_conf_thr
    matching_conf_thr = args.matching_conf_thr
    
    winsize = min(args.max_window_size, n_images)  # 0 < winsize <= n_images
    refid = min(n_images - 1, args.max_refid)  # 1 <= refid <= n_images - 1
    
    # shared_intrinsics = True if not use_calibrated_poses else False
    shared_intrinsics = True
    cam_size = 0.2
    mask_sky = False
    clean_depth = True

    fix_focal = args.fix_focal
    fix_pp = args.fix_principal_point
    fix_rotation = args.fix_rotation
    fix_translation = args.fix_translation

    lr1 = 0.07
    niter1 = args.n_coarse_iterations
    lr2 = 0.01
    niter2 = args.n_refinement_iterations  # TODO: Try 500
    optim_level = 'refine+depth'

    TSDF_thresh = args.TSDF_thresh

    output_conf_thr = args.output_conf_thr

    current_scene_state = None
    gradio_delete_cache = False  # TODO: Check if this is correct
    
    if args.output_dir is None:
        output_name = f'mast3r_allimages' if use_all_images else f'mast3r_{n_images}images'
        output_dir = os.path.join(scene_path, output_name)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # Use a unique temporary directory per process to avoid conflicts in multi-threading
    _outdir = os.path.join(output_dir, '.tmp_mast3r')
    os.makedirs(_outdir, exist_ok=True)

    print_debug("========== PARAMETERS ==========")
    print_debug(f"Scene path: {scene_path}")
    print_debug(f"Image directory: {image_dir}")
    print_debug(f"Output directory: {output_dir}")
    print_debug(f"Number of images: {n_images}")
    print_debug(f"Use all images: {use_all_images}")
    print_debug(f"Image size: {image_size}")
    print_debug(f"Scene graph type: {scenegraph_type}")
    print_debug(f"Minimum confidence threshold: {min_conf_thr}")
    print_debug(f"Matching confidence threshold: {matching_conf_thr}")
    print_debug(f"Window size: {winsize}")
    print_debug(f"Reference image id: {refid}")
    print_debug(f"Shared intrinsics: {shared_intrinsics}")
    print_debug(f"Camera size: {cam_size}")
    print_debug(f"Mask sky: {mask_sky}")
    print_debug(f"Clean depth: {clean_depth}")
    print_debug(f"Learning rate 1: {lr1}")
    print_debug(f"Number of coarse iterations: {niter1}")
    print_debug(f"Learning rate 2: {lr2}")
    print_debug(f"Number of refinement iterations: {niter2}")
    print_debug(f"Optimization level: {optim_level}")
    print_debug(f"TSDF threshold: {TSDF_thresh}")
    print_debug(f"Output confidence threshold: {output_conf_thr}")
    print_debug("================================\n")
    
    # TODO
    print_debug("\n[WARNING] This script needs to be updated to return data as float32.")
    
    # Helper functions
    class SparseGAState:
        def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
            self.sparse_ga = sparse_ga
            self.cache_dir = cache_dir
            self.outfile_name = outfile_name
            self.should_delete = should_delete

        def __del__(self):
            if not self.should_delete:
                return
            if self.cache_dir is not None and os.path.isdir(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            self.cache_dir = None
            if self.outfile_name is not None and os.path.isfile(self.outfile_name):
                os.remove(self.outfile_name)
            self.outfile_name = None

    def _build_first_frame_mapping(src_intrinsics_dict):
        """根据首帧的 src_intrinsics，构建从相机前缀到首帧文件名的映射。

        约定：文件名形如 cam00_xxxxx.jpg，其中最后一段为帧号。
        """
        mapping = {}
        for name in src_intrinsics_dict.keys():
            stem = Path(name).stem
            parts = stem.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                prefix = '_'.join(parts[:-1])
                frame_id = int(parts[-1])
            else:
                # 无法解析帧号，跳过
                continue

            if prefix not in mapping:
                mapping[prefix] = (frame_id, name)
            else:
                prev_frame, prev_name = mapping[prefix]
                if frame_id < prev_frame:
                    mapping[prefix] = (frame_id, name)
        print(mapping)
        # 只保留文件名
        return {k: v[1] for k, v in mapping.items()}

            
    def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                    cam_color=None, as_pointcloud=False,
                                    transparent_cams=False, silent=False):
        assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
        pts3d = to_numpy(pts3d)
        imgs = to_numpy(imgs)
        focals = to_numpy(focals)
        cams2world = to_numpy(cams2world)

        scene = trimesh.Scene()

        # full pointcloud
        if as_pointcloud:
            pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
            col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
            valid_msk = np.isfinite(pts.sum(axis=1))
            pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
            scene.add_geometry(pct)
        else:
            meshes = []
            for i in range(len(imgs)):
                pts3d_i = pts3d[i].reshape(imgs[i].shape)
                msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
                meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
            mesh = trimesh.Trimesh(**cat_meshes(meshes))
            scene.add_geometry(mesh)

        # add each camera
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(scene, pose_c2w, camera_edge_color,
                        None if transparent_cams else imgs[i], focals[i],
                        imsize=imgs[i].shape[1::-1], screen_width=cam_size)

        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
        if not silent:
            print('(exporting 3D scene to', outfile, ')')
        scene.export(file_obj=outfile)
        return outfile

    # ========== Start MASt3R-SfM ==========

    # Set device
    torch.cuda.set_device(args.gpu)
    device = torch.device(torch.cuda.current_device())

    # Load MASt3R model
    print('\nLoading MASt3R model...')
    torch.serialization.add_safe_globals([argparse.Namespace])
    model = AsymmetricMASt3R.from_pretrained(args.weights_path).to(device)
    chkpt_tag = hash_md5(args.weights_path)
    print('MASt3R model loaded.')
    
    # Load images
    print('\nLoading images...')
    all_image_names = np.sort(os.listdir(image_dir))
    if args.image_idx is not None:
        image_names = [all_image_names[i] for i in args.image_idx]
    elif use_all_images:
        image_names = all_image_names.tolist()
    else:
        # image_names = all_image_names[::len(all_image_names) // (n_images - 1)].tolist()  # Better
        image_names = [all_image_names[i * (len(all_image_names) // (n_images - 1))] for i in range(n_images)]
        # image_names = all_image_names[::math.ceil(len(all_image_names) / n_images)].tolist()  # Better

    filelist = [os.path.join(image_dir, image_name) for image_name in image_names]
    print('filelist:',filelist)
    silent = True
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    print(f'Loaded {len(imgs)} images.')
    print(f'Images have shape: {imgs[0]["img"].shape}')
    
    if not use_calibrated_poses:
        os.makedirs(f'{output_dir}/images', exist_ok=True)
        for idx_img, img_path in enumerate(filelist):
            img_fname = img_path.split('/')[-1]
            src_img = cv2.imread(img_path, 1)

            # ==========ADDED: Start by Cropping the original image==========
            print_debug(f"\nSource image has shape: {src_img.shape}")
            if True:
                # Get size of MASt3R images
                _height_resized, _width_resized = imgs[idx_img]['img'].size()[-2:]
                
                # Get original image size
                _height_original, _width_original = src_img.shape[:2]                
                
                # Get MASt3R image proportions
                _width_crop = int(round(max(_height_original, _width_original) / image_size * _width_resized))
                _height_crop = int(round(max(_height_original, _width_original) / image_size * _height_resized))
                
                print_debug(f"Width crop: {_width_crop}")
                print_debug(f"Height crop: {_height_crop}")
                
                # Update image and intrinsics
                print_debug(f"Source image shape (before cropping): {src_img.shape}")
                
                src_img = src_img[_height_original//2 - _height_crop//2:_height_original//2 + _height_crop//2, 
                                   _width_original//2 - _width_crop//2:_width_original//2 + _width_crop//2]
                print_debug(f"Source image shape (after cropping): {src_img.shape}")
                cv2.imwrite(f'{output_dir}/images/{img_fname}', src_img)
                filelist[idx_img] = f'{output_dir}/images/{img_fname}'
                
        imgs = load_images(filelist, size=image_size, verbose=not silent)
        print(f'Images have shape: {imgs[0]["img"].shape}')
    if not_first_frame:
        print("[INFO] Using first frame poses.")
        # 根据首帧的 COLMAP 结果构建「相机前缀 -> 首帧文件名」映射
        first_frame_map = _build_first_frame_mapping(src_intrinsics)

        # 为当前帧的每一张图找到对应相机在首帧中的文件名
        first_frame_image_names = []
        for img_name in image_names:
            stem = Path(img_name).stem
            parts = stem.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                cam_prefix = '_'.join(parts[:-1])
                if cam_prefix in first_frame_map:
                    ref_name = first_frame_map[cam_prefix]
                else:
                    # 前缀在首帧中找不到，退化为任意一个首帧文件名（保持行为鲁棒）
                    ref_name = next(iter(src_intrinsics.keys()))
                first_frame_image_names.append(ref_name)
            else:
                # 无法解析前缀，退化为任意一个首帧文件名
                ref_name = next(iter(src_intrinsics.keys()))
                first_frame_image_names.append(ref_name)

        intrinsics = np.stack([src_intrinsics[n] for n in first_frame_image_names], axis=0)
        extrinsics = np.stack([src_extrinsics[n] for n in first_frame_image_names], axis=0)
        calibrated_img_sizes = []

        print_debug('original intrinsics')
        print_debug(intrinsics)
        print_debug('original extrinsics')
        print_debug(extrinsics)

        # Modify images and intrinsics so that the principal points become the center of the images, before feeding them to MASt3R
        os.makedirs(f'{output_dir}/images', exist_ok=True)
        for idx_img, img_path in enumerate(filelist):
            print(img_path)
            img_fname = img_path.split('/')[-1]
            src_img = cv2.imread(img_path, 1)
            src_K = intrinsics[idx_img]
            
            # ==========ADDED==========
            src_height, src_width = src_img.shape[:2]
            src_dist = np.array([0., 0., 0., 0.]) # no distortion
            src_pp = src_K[:2,2]

            # Change the principal point so that it becomes the center of the image.
            tar_pp = np.array([
                int(min(src_pp[0], src_width - src_pp[0])), 
                int(min(src_pp[1], src_height - src_pp[1])), 
            ])
            
            print_debug(f"Tar pp: {tar_pp}")
            centered_height = 2 * tar_pp[1]
            centered_width = 2 * tar_pp[0]
            print_debug(f"Height centered: {centered_height}")
            print_debug(f"Width centered: {centered_width}")
            height_resized = imgs[idx_img]['img'].shape[-2]
            width_resized = imgs[idx_img]['img'].shape[-1]
            print_debug(f"Height resized: {height_resized}")
            print_debug(f"Width resized: {width_resized}")
            width_crop = int(round(max(centered_height, centered_width) / image_size * width_resized))
            height_crop = int(round(max(centered_height, centered_width) / image_size * height_resized))
            tar_pp = np.array([width_crop//2, height_crop//2])
            print_debug(f"New tar pp: {tar_pp}")
            # ==========END==========
            
            # Change the focals so that fx and fy become identical after resizing and cropping in load_images().
            w_, h_ = tar_pp * 2
            s_w = int(round(w_*image_size/max(w_, h_))) / w_
            s_h = int(round(h_*image_size/max(w_, h_))) / h_
            tar_focal = 0.5 * (src_K[0,0] + src_K[1,1])
            tar_focal_x = tar_focal * np.sqrt(s_w * s_h) / s_w
            tar_focal_y = tar_focal * np.sqrt(s_w * s_h) / s_h
            tar_K = src_K.copy()
            tar_K[:2,2] = tar_pp
            tar_K[0,0] = tar_focal_x
            tar_K[1,1] = tar_focal_y
            map1, map2 = cv2.initUndistortRectifyMap(
                src_K, src_dist, None, 
                tar_K, 2 * tar_pp,
                cv2.CV_32FC2
            )
            new_img = cv2.remap(
                src_img, 
                map1, map2, 
                interpolation=cv2.INTER_LANCZOS4, # cv2.INTER_LINEAR
                borderMode=cv2.BORDER_REPLICATE, # cv2.BORDER_CONSTANT
            )
            calibrated_img_sizes.append((new_img.shape[0], new_img.shape[1]))
            
            # ==========ADDED: Start by Cropping the original image==========
            print_debug(f"\nSource image has shape: {src_img.shape}")
            print_debug(f"Calibrated image has shape: {new_img.shape}")
            
            # ==========ADDED========== 
            
            print_debug(f"Source image has shape: {src_img.shape}")
            print_debug(f"Final image has shape: {new_img.shape}")
            cv2.imwrite(f'{output_dir}/images/{img_fname}', new_img)

            intrinsics[idx_img] = tar_K
            filelist[idx_img] = f'{output_dir}/images/{img_fname}'
        imgs = load_images(filelist, size=image_size, verbose=not silent)
        print_debug(f'Images have shape: {imgs[0]["img"].shape}')

        print_debug('intrinsics after cropping')
        print_debug(intrinsics)

        # Modify intrinsics according to MASt3R resizing
        for idx_img, img in enumerate(imgs):
            # Get size of MASt3R images
            height_resized, width_resized = img['img'].size()[-2:]
            
            # Load original rectified images to get the original size
            new_img = cv2.imread(filelist[idx_img], 1)
            height_original, width_original = new_img.shape[:2]
            # height_original, width_original = calibrated_img_sizes[idx_img]

            # Consider cropping inside load_images()!!!
            width_resized_simple = int(round(width_original*image_size/max(width_original, height_original)))
            height_resized_simple = int(round(height_original*image_size/max(width_original, height_original)))
            ofs_u = width_resized_simple//2 - width_resized//2
            ofs_v = height_resized_simple//2 - height_resized//2

            # Scaling focals and principal points
            intrinsics[idx_img][0,:] *= (width_resized_simple / width_original)
            intrinsics[idx_img][1,:] *= (height_resized_simple / height_original)
            # Adjusting principal points. Centered cropping, so the principal point is still centered
            intrinsics[idx_img][0,2] -= ofs_u  
            intrinsics[idx_img][1,2] -= ofs_v
            
            
        imgs = load_images(filelist, size=image_size, verbose=not silent)
        print_debug(f'Images have shape: {imgs[0]["img"].shape}')
        
        print_debug('intrinsics after resizing')
        print_debug(intrinsics)
    if use_calibrated_poses and not not_first_frame:
        print("[INFO] Using calibrated poses.")
        intrinsics = np.stack([src_intrinsics[img_name] for img_name in image_names], axis=0)
        extrinsics = np.stack([src_extrinsics[img_name] for img_name in image_names], axis=0)
        calibrated_img_sizes = []

        print_debug('original intrinsics')
        print_debug(intrinsics)
        print_debug('original extrinsics')
        print_debug(extrinsics)

        # Modify images and intrinsics so that the principal points become the center of the images, before feeding them to MASt3R
        os.makedirs(f'{output_dir}/images', exist_ok=True)
        for idx_img, img_path in enumerate(filelist):
            img_fname = img_path.split('/')[-1]
            src_img = cv2.imread(img_path, 1)
            src_K = intrinsics[idx_img]
            
            # ==========ADDED==========
            src_height, src_width = src_img.shape[:2]
            src_dist = np.array([0., 0., 0., 0.]) # no distortion
            src_pp = src_K[:2,2]

            # Change the principal point so that it becomes the center of the image.
            tar_pp = np.array([
                int(min(src_pp[0], src_width - src_pp[0])), 
                int(min(src_pp[1], src_height - src_pp[1])), 
            ])
            
            print_debug(f"Tar pp: {tar_pp}")
            centered_height = 2 * tar_pp[1]
            centered_width = 2 * tar_pp[0]
            print_debug(f"Height centered: {centered_height}")
            print_debug(f"Width centered: {centered_width}")
            height_resized = imgs[idx_img]['img'].shape[-2]
            width_resized = imgs[idx_img]['img'].shape[-1]
            print_debug(f"Height resized: {height_resized}")
            print_debug(f"Width resized: {width_resized}")
            width_crop = int(round(max(centered_height, centered_width) / image_size * width_resized))
            height_crop = int(round(max(centered_height, centered_width) / image_size * height_resized))
            tar_pp = np.array([width_crop//2, height_crop//2])
            print_debug(f"New tar pp: {tar_pp}")
            # ==========END==========
            
            # Change the focals so that fx and fy become identical after resizing and cropping in load_images().
            w_, h_ = tar_pp * 2
            s_w = int(round(w_*image_size/max(w_, h_))) / w_
            s_h = int(round(h_*image_size/max(w_, h_))) / h_
            tar_focal = 0.5 * (src_K[0,0] + src_K[1,1])
            tar_focal_x = tar_focal * np.sqrt(s_w * s_h) / s_w
            tar_focal_y = tar_focal * np.sqrt(s_w * s_h) / s_h
            tar_K = src_K.copy()
            tar_K[:2,2] = tar_pp
            tar_K[0,0] = tar_focal_x
            tar_K[1,1] = tar_focal_y
            map1, map2 = cv2.initUndistortRectifyMap(
                src_K, src_dist, None, 
                tar_K, 2 * tar_pp,
                cv2.CV_32FC2
            )
            new_img = cv2.remap(
                src_img, 
                map1, map2, 
                interpolation=cv2.INTER_LANCZOS4, # cv2.INTER_LINEAR
                borderMode=cv2.BORDER_REPLICATE, # cv2.BORDER_CONSTANT
            )
            calibrated_img_sizes.append((new_img.shape[0], new_img.shape[1]))
            
            # ==========ADDED: Start by Cropping the original image==========
            print_debug(f"\nSource image has shape: {src_img.shape}")
            print_debug(f"Calibrated image has shape: {new_img.shape}")
            
            # ==========ADDED========== 
            
            print_debug(f"Source image has shape: {src_img.shape}")
            print_debug(f"Final image has shape: {new_img.shape}")
            cv2.imwrite(f'{output_dir}/images/{img_fname}', new_img)

            intrinsics[idx_img] = tar_K
            filelist[idx_img] = f'{output_dir}/images/{img_fname}'
        imgs = load_images(filelist, size=image_size, verbose=not silent)
        print_debug(f'Images have shape: {imgs[0]["img"].shape}')

        print_debug('intrinsics after cropping')
        print_debug(intrinsics)

        # Modify intrinsics according to MASt3R resizing
        for idx_img, img in enumerate(imgs):
            # Get size of MASt3R images
            height_resized, width_resized = img['img'].size()[-2:]
            
            # Load original rectified images to get the original size
            new_img = cv2.imread(filelist[idx_img], 1)
            height_original, width_original = new_img.shape[:2]
            # height_original, width_original = calibrated_img_sizes[idx_img]

            # Consider cropping inside load_images()!!!
            width_resized_simple = int(round(width_original*image_size/max(width_original, height_original)))
            height_resized_simple = int(round(height_original*image_size/max(width_original, height_original)))
            ofs_u = width_resized_simple//2 - width_resized//2
            ofs_v = height_resized_simple//2 - height_resized//2

            # Scaling focals and principal points
            intrinsics[idx_img][0,:] *= (width_resized_simple / width_original)
            intrinsics[idx_img][1,:] *= (height_resized_simple / height_original)
            # Adjusting principal points. Centered cropping, so the principal point is still centered
            intrinsics[idx_img][0,2] -= ofs_u  
            intrinsics[idx_img][1,2] -= ofs_v
            
            # ==========ADDED: Save Cropped Image==========
            if True and False:
                from torchvision.transforms.functional import center_crop
                # Get MASt3R image proportions
                width_crop = int(round(max(height_original, width_original) / image_size * width_resized))
                height_crop = int(round(max(height_original, width_original) / image_size * height_resized))
                
                print_debug(f"Width crop: {width_crop}")
                print_debug(f"Height crop: {height_crop}")
                if True:
                    cropped_img = center_crop(
                        torch.tensor(new_img, device=device).permute(2, 0, 1),
                        output_size=(height_crop, width_crop)
                    ).permute(1, 2, 0).cpu().numpy()
                else:
                    src_img = src_img[_height_original//2 - _height_crop//2:_height_original//2 + _height_crop//2, 
                                   _width_original//2 - _width_crop//2:_width_original//2 + _width_crop//2]
                cv2.imwrite(filelist[idx_img], cropped_img)
                print_debug(f"Image saved to {filelist[idx_img]}")
                
                # Update image and intrinsics
                print_debug(f"Cropped image shape: {cropped_img.shape}")       
                
                new_resized_shape = load_images([filelist[idx_img]], size=image_size, verbose=False)[0]['img'][0].shape
                print_debug(f"New resized shape: {new_resized_shape}")
                if new_resized_shape[-2] != height_resized or new_resized_shape[-1] != width_resized:
                    raise ValueError(f"Cropped image shape {new_resized_shape} does not match MASt3R resized shape {(height_resized, width_resized)}")
            # ==========END==========

        imgs = load_images(filelist, size=image_size, verbose=not silent)
        print_debug(f'Images have shape: {imgs[0]["img"].shape}')
        
        print_debug('intrinsics after resizing')
        print_debug(intrinsics)

    # Load retrieval model
    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert args.retrieval_model is not None
        retriever = Retriever(args.retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)
        # Cleanup
        del retriever
        torch.cuda.empty_cache()
        
    # Build scene graph
    print('\nStart building scene graph...')
    t0 = time.time()
    
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=_outdir)
    else:
        cache_dir = os.path.join(_outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    print('Done building scene graph.')
    
    # Run sparse global alignment
    print('Start global alignment...')
    kw = {
        'init': {},
        'opt_pp': True if not fix_pp else False,
        'opt_focal': True if not fix_focal else False,
        'opt_quat': True if not fix_rotation else False,
        'opt_tran': True, # if not fix_translation else False, # For now, this does not work well.
        'opt_size': True, # if not fix_translation else False, # Instead, we do post alignment.
    }
    if use_calibrated_poses and not not_first_frame:
        for idx_img, img_path in enumerate(filelist):
            kw['init'][img_path] = {
                'intrinsics': torch.from_numpy(intrinsics[idx_img].astype(np.float32)).to(device),
                'cam2w': torch.inverse(torch.from_numpy(extrinsics[idx_img].astype(np.float32)).to(device)),
            }
            
    if not_first_frame:
        kw = {
            'init': {},
            'opt_pp': False ,
            'opt_focal': False ,
            'opt_quat': False ,
            'opt_tran': True, # if not fix_translation else False, # For now, this does not work well.
            'opt_size': True, # if not fix_translation else False, # Instead, we do post alignment.
        }
        for idx_img, img_path in enumerate(filelist):
            kw['init'][img_path] = {
                'intrinsics': torch.from_numpy(intrinsics[idx_img].astype(np.float32)).to(device),
                'cam2w': torch.inverse(torch.from_numpy(extrinsics[idx_img].astype(np.float32)).to(device)),
            }

    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)  # TODO: Check that removing kw is fine
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=_outdir)
    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    print('Done global alignment.')
    print(f'Optimization complete in {time.time() - t0:.2f}s.')
    
    if scene_state is None:
        print('No scene state')
    outfile = scene_state.outfile_name
    if outfile is None:
        print('No outfile')

    # Get optimized values from scene
    print('Saving optimized scene...')
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    pps = scene.get_principal_points().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        print('TSDF post-processing...')
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))



    if not_first_frame:
        print('[INFO] Aligning camera locations...')
        calib_cam_locations = []
        est_cam_locations = []
        for idx_view, img_name in enumerate(image_names):
            # 使用与上面相同的首帧映射，而不是硬编码 0001.jpg
            first_frame_map = _build_first_frame_mapping(src_intrinsics)
            stem = Path(img_name).stem
            parts = stem.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                cam_prefix = '_'.join(parts[:-1])
                if cam_prefix in first_frame_map:
                    ref_name = first_frame_map[cam_prefix]
                else:
                    ref_name = next(iter(src_extrinsics.keys()))
            else:
                ref_name = next(iter(src_extrinsics.keys()))

            calib_cam_locations.append(np.linalg.inv(src_extrinsics[ref_name])[:3,3])
            est_cam_locations.append(cams2world[idx_view][:3,3].cpu().numpy())
        calib_cam_locations = np.stack(calib_cam_locations, axis=0)
        est_cam_locations = np.stack(est_cam_locations, axis=0)

        if True:#fix_rotation:
            # Modify only scale and translation
            x = est_cam_locations - np.mean(est_cam_locations, axis=0, keepdims=True)
            y = calib_cam_locations - np.mean(calib_cam_locations, axis=0, keepdims=True)
            # minimize ||y - scale * x||^2 w.r.t scale
            # scale = \sum{xy} / \sum{x^2}
            scale_est2calib = np.sum(x * y) / np.sum(x**2)
            ofs_est2calib = np.mean(calib_cam_locations, axis=0) - scale_est2calib * np.mean(est_cam_locations, axis=0)

            mean_residual = np.mean(np.linalg.norm(scale_est2calib * est_cam_locations + ofs_est2calib - calib_cam_locations, axis=-1))
            print_debug('Scale and translation offsets:', scale_est2calib, ofs_est2calib)
            print_debug('Mean residual of camera locations:', mean_residual)

            # Modify camera poses and 3D points
            cams2world[:,:3,3] = scale_est2calib * cams2world[:,:3,3] + torch.from_numpy(ofs_est2calib).to(cams2world.device)
            print_debug('modified cams2world', torch.inverse(cams2world).numpy())
            for i_ in range(len(pts3d)):
                pts3d[i_] = scale_est2calib * pts3d[i_] + ofs_est2calib

        else:
            # Modify rotation, scale, and translaion
            raise NotImplementedError()
        cams2world = torch.from_numpy(np.linalg.inv(extrinsics))
       
        print_debug('output extrinsics:')
        print_debug(torch.inverse(cams2world).numpy())

        # Unnormalize world scale and translaion for data with cameras.npz
        if src_scale_mats is not None:
            for i_ in range(len(pts3d)):
                S = src_scale_mats[image_names[i_]]
                cam_location = cams2world[i_:i_+1,:3,3].cpu().numpy()
                new_cam_location = (S[:3,:3] @ cam_location.T + S[:3,3:4]).T
                cams2world[i_:i_+1,:3,3] = torch.from_numpy(new_cam_location).to(cams2world.device)
                pts3d[i_] = (S[:3,:3] @ pts3d[i_].T + S[:3,3:4]).T
    # Align the recovered camera locations with calibrated ones
    if use_calibrated_poses and args.align_camera_locations and not not_first_frame:
        print('[INFO] Aligning camera locations...')
        calib_cam_locations = []
        est_cam_locations = []
        for idx_view, img_name in enumerate(image_names):
            calib_cam_locations.append(np.linalg.inv(src_extrinsics[img_name])[:3,3])
            est_cam_locations.append(cams2world[idx_view][:3,3].cpu().numpy())
        calib_cam_locations = np.stack(calib_cam_locations, axis=0)
        est_cam_locations = np.stack(est_cam_locations, axis=0)

        if True:#fix_rotation:
            # Modify only scale and translation
            x = est_cam_locations - np.mean(est_cam_locations, axis=0, keepdims=True)
            y = calib_cam_locations - np.mean(calib_cam_locations, axis=0, keepdims=True)
            # minimize ||y - scale * x||^2 w.r.t scale
            # scale = \sum{xy} / \sum{x^2}
            scale_est2calib = np.sum(x * y) / np.sum(x**2)
            ofs_est2calib = np.mean(calib_cam_locations, axis=0) - scale_est2calib * np.mean(est_cam_locations, axis=0)

            mean_residual = np.mean(np.linalg.norm(scale_est2calib * est_cam_locations + ofs_est2calib - calib_cam_locations, axis=-1))
            print_debug('Scale and translation offsets:', scale_est2calib, ofs_est2calib)
            print_debug('Mean residual of camera locations:', mean_residual)

            # Modify camera poses and 3D points
            cams2world[:,:3,3] = scale_est2calib * cams2world[:,:3,3] + torch.from_numpy(ofs_est2calib).to(cams2world.device)
            for i_ in range(len(pts3d)):
                pts3d[i_] = scale_est2calib * pts3d[i_] + ofs_est2calib

        else:
            # Modify rotation, scale, and translaion
            raise NotImplementedError()

        if fix_rotation and fix_translation:
            cams2world = torch.from_numpy(np.linalg.inv(extrinsics))
        elif fix_translation:
            cams2world[:,:3,3] = torch.from_numpy(np.linalg.inv(extrinsics))[:,:3,3]
        elif fix_rotation:
            cams2world[:,:3,:3] = torch.from_numpy(np.linalg.inv(extrinsics))[:,:3,:3]

        print_debug('output extrinsics:')
        print_debug(torch.inverse(cams2world).numpy())

        # Unnormalize world scale and translaion for data with cameras.npz
        if src_scale_mats is not None:
            for i_ in range(len(pts3d)):
                S = src_scale_mats[image_names[i_]]
                cam_location = cams2world[i_:i_+1,:3,3].cpu().numpy()
                new_cam_location = (S[:3,:3] @ cam_location.T + S[:3,3:4]).T
                cams2world[i_:i_+1,:3,3] = torch.from_numpy(new_cam_location).to(cams2world.device)
                pts3d[i_] = (S[:3,:3] @ pts3d[i_].T + S[:3,3:4]).T
    
    # Save cameras    
    output_cameras = {
        'filepaths': [img['instance'] for img in imgs],
        'focals': focals.numpy().tolist(),
        'cams2world': cams2world.numpy().tolist(),
    }
    with open(os.path.join(output_dir, 'cameras.json'), 'w') as f:
        json.dump(output_cameras, f)
    

    cameras_colmap = {}
    images_colmap = {}
    points3d_colmap = {}
    points3d_id_ofs = 1
    all_xyzs = []
    all_rgbs = []
    for idx_img in range(len(imgs)):
        camera_id = idx_img + 1

        # intrinsics
        height_original, width_original = cv2.imread(imgs[idx_img]['instance'],0).shape[:2]
        width_resized = int(imgs[idx_img]['img'].size(-1))
        height_resized = int(imgs[idx_img]['img'].size(-2))

        # Consider cropping inside load_images()!!!
        width_resized_simple = int(round(width_original*image_size/max(width_original, height_original)))
        height_resized_simple = int(round(height_original*image_size/max(width_original, height_original)))
        ofs_u = width_resized_simple//2 - width_resized//2
        ofs_v = height_resized_simple//2 - height_resized//2

        scale_w = width_original / width_resized_simple
        scale_h = height_original / height_resized_simple
        cameras_colmap[camera_id] = Camera(
            int(camera_id), 
            'PINHOLE', 
            width=width_original, 
            height=height_original, 
            params=np.array([
                focals[idx_img].item() * scale_w,
                focals[idx_img].item() * scale_h,
                (pps[idx_img][0].item() + ofs_u) * scale_w,
                (pps[idx_img][1].item() + ofs_v) * scale_h,
            ])
        )

        # extrinsics
        w2c = np.linalg.inv(cams2world[idx_img].numpy())
        print_debug(w2c)
        # points
        pixels = np.mgrid[:width_resized, :height_resized].T.reshape(-1, 2)
        pixels_original = pixels * np.array([scale_w, scale_h])
        point3d_ids = []
        xys = []
        img_original = cv2.imread(imgs[idx_img]['instance'],1)[...,::-1]
        # TODO: avoid slow for-loop
        for idx_pt2d, (xyz, pt2d) in enumerate(zip(pts3d[idx_img], pixels_original)):
            pixel_conf = confs[idx_img][pixels[idx_pt2d][1], pixels[idx_pt2d][0]]
            if pixel_conf < output_conf_thr: # ignore points with low confidence
                continue

            point3d_id = points3d_id_ofs + idx_pt2d
            xys.append(pt2d)
            point3d_ids.append(point3d_id)

            rgb = img_original[int(np.clip(pt2d[1],0,height_original)), int(np.clip(pt2d[0],0,width_original))]
            error = np.array(2., dtype=np.float64) # TODO: use actual reprojection error?

            image_ids = np.array([camera_id,], dtype=np.int64)
            point2D_idxs=np.array([idx_pt2d,], dtype=np.int64)

            points3d_colmap[point3d_id] = Point3D(
                id=point3d_id,
                xyz=xyz.astype(np.float64),
                rgb=rgb.astype(np.int64),
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
            all_xyzs.append(xyz.astype(np.float64))
            all_rgbs.append(rgb.astype(np.float64) / 255.)
        points3d_id_ofs += len(pts3d[idx_img])

        images_colmap[camera_id] = Image(
            camera_id, # id
            rotmat2qvec(w2c[:3,:3]).astype(np.float64), # qvec
            w2c[:3,3].astype(np.float64), # tvec
            camera_id, # camera_model_id
            imgs[idx_img]['instance'].split('/')[-1], # name
            np.array(xys, dtype=np.float64), # xys
            np.array(point3d_ids, dtype=np.int64) # point3D_ids
        )

    os.makedirs(f'{output_dir}/sparse/0', exist_ok=True)
    write_cameras_binary(cameras_colmap, f'{output_dir}/sparse/0/cameras.bin')
    write_images_binary(images_colmap, f'{output_dir}/sparse/0/images.bin')
    write_points3D_binary(points3d_colmap, f'{output_dir}/sparse/0/points3D.bin')
    if True:
        write_cameras_text(cameras_colmap, f'{output_dir}/sparse/0/cameras.txt')
        write_images_text(images_colmap, f'{output_dir}/sparse/0/images.txt')
        write_points3D_text(points3d_colmap, f'{output_dir}/sparse/0/points3D.txt')

    # save 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack(all_xyzs, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.stack(all_rgbs, axis=0))
    o3d.io.write_point_cloud(f"{output_dir}/points.ply", pcd)

    # Save pointmaps
    pointmaps_dir = os.path.join(output_dir, 'pointmaps')
    os.makedirs(pointmaps_dir, exist_ok=True)
    for i_image in range(len(imgs)):
        img_path = imgs[i_image]['instance']
        img_name = os.path.basename(img_path).split('.')[0]
        output_pointmap = {
            'rgb': None if use_all_images else rgbimg[i_image].tolist(),
            'points': pts3d[i_image].tolist(),
            'confs': confs[i_image].tolist(),
        }
        with open(os.path.join(pointmaps_dir, f'{img_name}.json'), 'w') as f:
            json.dump(output_pointmap, f)
        
    if args.save_glb:
        as_pointcloud = True
        transparent_cams = False
        msk = to_numpy([c > min_conf_thr for c in confs])
        _convert_scene_output_to_glb(
            # outfile, 
            os.path.join(output_dir, 'scene.glb'),
            rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
            transparent_cams=transparent_cams, cam_size=cam_size, silent=silent
        )
    print('Scene saved.')
    
    # Remove temporary files
    print('\nCleaning up temporary files...')
    shutil.rmtree(_outdir)
    print('Temporary files cleaned up.')
    
    print('\nInitial SfM complete.')
