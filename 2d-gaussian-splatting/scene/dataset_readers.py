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
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
#

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    charts_priors: dict = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    # sort extrinsics by name for time-based processing
    sorted_extrinsics = sorted(cam_extrinsics.values(), key=lambda x: x.name)
    num_cameras = len(sorted_extrinsics)

    for idx, extr in enumerate(sorted_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(sorted_extrinsics)))
        sys.stdout.flush()

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # time is normalized to [0, 1]
        time = idx / (num_cameras - 1) if num_cameras > 1 else 0.5

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=time)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    #normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # 如果没有normals，就不要normals返回即可，不要报错
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        return BasicPointCloud(points=positions, colors=colors, normals=normals)
    except (KeyError, ValueError):
        # 没有 normals 字段，只返回 points 和 colors
        return BasicPointCloud(points=positions, colors=colors, normals=None)
    #return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))

    if eval:
        train_cam_infos = [c for i, c in enumerate(cam_infos) if i % llffhold != 0]
        test_cam_infos = [c for i, c in enumerate(cam_infos) if i % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # 加载charts priors
    charts_priors = None
    charts_priors_path = os.path.join(path, "preprocessed_priors", "charts_priors.npz")
    if os.path.exists(charts_priors_path):
        import torch
        import numpy as np
        from matcha.dm_scene.charts import build_priors_from_charts_data
        charts_data = dict(np.load(charts_priors_path, allow_pickle=True))
        charts_priors = build_priors_from_charts_data(charts_data, train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           charts_priors=charts_priors)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], time=0))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    charts_priors = None
    charts_priors_path = os.path.join(path, "preprocessed_priors", "charts_priors.npz")
    if os.path.exists(charts_priors_path):
        import torch
        import numpy as np
        from matcha.dm_scene.charts import build_priors_from_charts_data
        charts_data = dict(np.load(charts_priors_path, allow_pickle=True))
        charts_priors = build_priors_from_charts_data(charts_data, train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           charts_priors=charts_priors)
    return scene_info


def readMultipleViewinfos(path, images, eval, llffhold=8):
    # Import 4D-style multi-view frame loader
   
    from scene.multipleview_dataset import multipleview_dataset
    #print("????")
    # Load train/test camera sequences with per-frame images
    cam_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cam_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cam_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cam_intrinsic_file)
    # Load dataset with camera-based splitting
    print("[INFO] Loading multi-view datasets with camera splitting...")

    # Create dataset instances with camera splitting
    train_dataset = multipleview_dataset(cam_extrinsics, cam_intrinsics, path, split="train")
    test_dataset = multipleview_dataset(cam_extrinsics, cam_intrinsics, path, split="test")

    train_cams = []
    print("[INFO] Converting training dataset to CameraInfo objects...")
    for idx in range(len(train_dataset)):
        _, (R, T), time = train_dataset[idx]
        image_path = train_dataset.image_paths[idx]
        try:
            # Load PIL image
            with Image.open(image_path) as img:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                #frame_seq = int(image_name.split("_")[2])
                #print(frame_seq)
                width, height = img.width, img.height
                train_cams.append(CameraInfo(uid=idx, R=R, T=T,
                                                FovY=train_dataset.FovY, FovX=train_dataset.FovX,
                                                image=img.copy(), image_path=image_path,
                                                image_name=image_name, width=width,
                                                height=height, time=time))
        except Exception as e:
            print(f"[WARNING] Failed to load training image {image_path}: {e}, skipping")
            continue

    test_cams = []
    print("[INFO] Converting test dataset to CameraInfo objects...")
    # Fixed: Use test_dataset instead of train_dataset for poses
    for idx in range(len(test_dataset)):
        _, (R, T), time = test_dataset[idx]
        image_path = test_dataset.image_paths[idx]
        try:
            # Load PIL image
            with Image.open(image_path) as img:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                width, height = img.width, img.height
                test_cams.append(CameraInfo(uid=idx, R=R, T=T,
                                                FovY=test_dataset.FovY, FovX=test_dataset.FovX,
                                                image=img.copy(), image_path=image_path,
                                                image_name=image_name, width=width,
                                                height=height, time=time))
        except Exception as e:
            print(f"[WARNING] Failed to load test image {image_path}: {e}, skipping")
            continue
        # img, (R, T), time = test_dataset[idx]
        # try:
        #     image_np = img.permute(1,2,0).cpu().numpy()
        # except:
        #     image_np = np.array(img)
        # image_path = test_dataset.image_paths[idx]
        # image_name = os.path.splitext(os.path.basename(image_path))[0]
        # height, width = image_np.shape[0], image_np.shape[1]
        # test_cams.append(CameraInfo(uid=idx, R=R, T=T,
        #                             FovY=test_dataset.FovY, FovX=test_dataset.FovX,
        #                             image=image_np, image_path=image_path,
        #                             image_name=image_name, width=width,
        #                             height=height, time=time))
    # Compute normalization based on train cameras
    nerf_normalization = getNerfppNorm(train_cams)
    # Read or convert point cloud for multiple view
    ply_path = os.path.join(path, "points3D_multipleview.ply")
    bin_path = os.path.join(path, "points3D_multipleview.bin")
    txt_path = os.path.join(path, "points3D_multipleview.txt")
    if not os.path.exists(ply_path):
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)
    # Return scene info with ply and normalization
    charts_priors = None
    charts_priors_path = os.path.join(path, "preprocessed_priors", "charts_priors.npz")
    if os.path.exists(charts_priors_path):
        import torch
        import numpy as np
        from matcha.dm_scene.charts import build_priors_from_charts_data
        charts_data = dict(np.load(charts_priors_path, allow_pickle=True))
        charts_priors = build_priors_from_charts_data(charts_data, train_cams)
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cams,
        test_cameras=test_cams,
        video_cameras=test_dataset.video_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        charts_priors=charts_priors
    )
# 注册动态多视图回调
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "MultipleView": readMultipleViewinfos
}
 