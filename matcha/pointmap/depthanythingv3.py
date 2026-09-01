import os
import sys
import torch
from torch.nn.functional import normalize as torch_normalize
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
)
import open3d as o3d

# DepthAnything V3
from depth_anything_3.api import DepthAnything3

# Custom code
from matcha.pointmap.base import PointMap
from matcha.pointmap.utils import load_colmap_scene
try:
    from matcha.pointmap.dust3r import compute_dust3r_scene
except:
    print("Dust3R not found.")
from matcha.pointmap.mast3r import compute_mast3r_scene
from matcha.dm_scene.cameras import CamerasWrapper, P3DCameras
from matcha.dm_utils.rendering import fov2focal

def export_pointmap_to_pcd(pointmap, save_path="aligned_pointmap.ply"):
    all_points = []
    all_colors = []

    for img, pts3d in zip(pointmap.images, pointmap.points3d):
        mask = torch.isfinite(pts3d[..., 0])
        pts = pts3d[mask].view(-1, 3).cpu().numpy()
        colors = img[mask].view(-1, 3).cpu().numpy()
        all_points.append(pts)
        all_colors.append(colors)

    all_points = torch.tensor(np.concatenate(all_points, axis=0))
    all_colors = torch.tensor(np.concatenate(all_colors, axis=0))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(all_colors.numpy())
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"✅ Exported pointmap to {save_path}")
    # Calculate normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)

    # Optionally, you can save the normals along with the point cloud
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)
    print(f"✅ Exported pointmap with normals to {save_path}")


class PointMapDepthAnything(PointMap):
    def __init__(
        self,
        scene_cameras:CamerasWrapper,
        scene_eval_cameras:CamerasWrapper=None,
        **kwargs,
    ):
        super(PointMapDepthAnything, self).__init__(**kwargs)
        self.scene_cameras = scene_cameras
        self.scene_eval_cameras = scene_eval_cameras


def load_model(
    model_name='depth-anything/DA3NESTED-GIANT-LARGE',
    device='cpu',
):
    """
    Load Depth Anything V3 model.

    Args:
        model_name: HuggingFace model name for Depth Anything V3
        device: Device to load the model on

    Returns:
        model: Loaded Depth Anything V3 model
    """
    print(f"[INFO] Loading Depth Anything V3 model: {model_name}")
    #DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to("cuda")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(device).eval()
    return model


@torch.no_grad()
def apply_depthanything(
    model:DepthAnything3,
    image:torch.Tensor
):
    """
    Apply Depth Anything V3 to get depth prediction for a single image.

    Args:
        model (DepthAnything3): Depth Anything V3 model
        image (torch.Tensor): Has shape (H, W, 3) and RGB format. Values are between 0 and 1.

    Returns:
        torch.Tensor: Disparity map (inverse depth) of shape (H, W)
    """
    # Convert from RGB [0, 1] tensor to RGB [0, 255] numpy array
    input_image = (image * 255).int().cpu().numpy().astype(np.uint8)

    # V3 expects a list of image paths or PIL images, but we can work around this
    # by saving to a temporary file or using PIL
    from PIL import Image
    import tempfile

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(input_image)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        pil_image.save(tmp_path)

    try:
        # Run inference - V3 returns a prediction object with .depth attribute
        prediction = model.inference([tmp_path])

        # Extract depth from prediction (shape: (1, H, W))
        depth = prediction.depth[0]  # Get first (and only) image's depth

        # Convert to tensor if it's not already
        if not isinstance(depth, torch.Tensor):
            depth = torch.from_numpy(depth).to(image.device)
        elif depth.device != image.device:
            depth = depth.to(image.device)

        # Convert depth to disparity (inverse depth)
        # Depth Anything V3 returns metric depth, we need disparity for compatibility
        disp = 1.0 / (depth + 1e-6)

    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

    return disp


def get_points_depth_in_depthmap(
    pts:torch.Tensor,
    depthmap:torch.Tensor,
    p3d_camera:P3DCameras,
    padding_mode:str='zeros',  #'border'
):
    """_summary_

    Args:
        pts (torch.Tensor): Has shape (N, 3).
        depthmap (torch.Tensor): Has shape (H, W) or (H, W, 1).
        p3d_camera (P3DCameras): _description_

    Returns:
        _type_: _description_
    """
    image_height, image_width = depthmap.shape[:2]

    depth_view = depthmap.reshape(1, image_height, image_width, 1).permute(0, 3, 1, 2)
    pts_projections = p3d_camera.get_full_projection_transform().transform_points(pts)

    factor = -1 * min(image_height, image_width)
    pts_projections[..., 0] = factor / image_width * pts_projections[..., 0]
    pts_projections[..., 1] = factor / image_height * pts_projections[..., 1]
    pts_projections = pts_projections[..., :2].view(1, -1, 1, 2)

    map_z = torch.nn.functional.grid_sample(
        input=depth_view,
        grid=pts_projections,
        mode='bilinear',
        padding_mode=padding_mode,  # 'reflection', 'zeros'
    )
    map_z = map_z[0, 0, :, 0]
    fov_mask = map_z > 0.

    return map_z, fov_mask


def get_points_color_in_image(
    pts:torch.Tensor,
    image:torch.Tensor,
    p3d_camera:P3DCameras,
):
    """_summary_

    Args:
        pts (torch.Tensor): Has shape (N, 3).
        image (torch.Tensor): Has shape (H, W, 3).
        p3d_camera (P3DCameras): _description_

    Returns:
        _type_: _description_
    """
    image_height, image_width = image.shape[:2]

    image_view = image.unsqueeze(0).permute(0, 3, 1, 2)
    pts_projections = p3d_camera.get_full_projection_transform().transform_points(pts)

    factor = -1 * min(image_height, image_width)
    pts_projections[..., 0] = factor / image_width * pts_projections[..., 0]
    pts_projections[..., 1] = factor / image_height * pts_projections[..., 1]
    pts_projections = pts_projections[..., :2].view(1, -1, 1, 2)

    image_colors = torch.nn.functional.grid_sample(
        input=image_view,
        grid=pts_projections,
        mode='bilinear',
        padding_mode='zeros',  # 'reflection', 'zeros'
    )
    image_colors = image_colors[0, :, :, 0]
    image_colors = image_colors.transpose(-1, -2)
    return image_colors


def fit_depth_to_point_cloud(
    disp:torch.Tensor,
    pts:torch.Tensor,
    training_cameras:CamerasWrapper,
    camera_idx:int,
    image:torch.Tensor=None,
    pt_colors:torch.Tensor=None,
    return_alpha_beta:bool=False,
    use_rasterizer:bool=False,
    use_fov_mask:bool=False,
):
    """Computes an affine transformation of the depth to match the depth of the projections of the points in pts.
    To do this, we minimize the L2 distance between the true depth of the points and the depth of the corresponding projections in the depth map.
    Color features from images can be used for weighting the loss and enforce color consistency.

    ---Details---
    We want sum( weights * (alpha + beta * depthmap_points_depth - true_points_depth) ** 2) ~ 0, i.e.
    sum (weights * (alpha ** 2 + true_points_depth ** 2 + beta ** 2 * depthmap_points_depth ** 2
       + 2 * alpha * beta * depthmap_points_depth - 2 * alpha * true_points_depth - 2 * beta * true_points_depth * depthmap_points_depth ))

    Derivative wrt alpha: 2 * sum(weights) * alpha + 2 * beta * sum(weights * depthmap_points_depth) - 2 * sum(weights * true_points_depth)
    Derivative wrt beta: 2 * beta * sum(weights * depthmap_points_depth**2) + 2 * alpha * sum(weights * depthmap_points_depth) - 2 * sum(weights * true_points_depth * depthmap_points_depth)

    We want the derivative to be = 0, which implies:

    alpha = (- 2 * beta * sum(weights * depthmap_points_depth) + 2 * sum(weights * true_points_depth)) / (2 * sum(weights)) so that
    2 * beta * sum(weights * depthmap_points_depth**2) + 2 * (- beta * sum(weights * depthmap_points_depth)
       + sum(weights * true_points_depth)) / sum(weights) * sum(weights * depthmap_points_depth)
       - 2 * sum(weights * true_points_depth * depthmap_points_depth) = 0

    This gives the following:
    (1) beta = beta_num / beta_denom
    where
    (1a) beta_num = sum(weights * true_points_depth * depthmap_points_depth) - sum(weights * true_points_depth) * sum(weights * depthmap_points_depth) / sum(weights)
    (1b) beta_denom = sum(weights * depthmap_points_depth ** 2) - sum(weights * depthmap_points_depth) ** 2 / sum(weights)
    and
    (2) alpha = sum(weights * (true_points_depth - beta * depthmap_points_depth)) / sum(weights)

    Args:
        depth (torch.Tensor): Has shape (H, W).
        pts (torch.Tensor): Has shape (N, 3).
        training_cameras (CamerasWrapper): _description_
        camera_idx (int): Should be in range(len(training_cameras)).
        image (torch.Tensor, optional): Has shape (H, W, 3). Defaults to None.
        pt_colors (torch.Tensor, optional): Has shape (N, 3). Defaults to None.
        return_alpha_beta (bool, optional): Defaults to False.
        use_rasterizer (bool, optional): Defaults to False.

    Returns:
        _type_: _description_
    """
    assert camera_idx < len(training_cameras)
    p3d_camera = training_cameras.p3d_cameras[camera_idx]

    if use_rasterizer:
        raise NotImplementedError("Rasterizer implementation is not really satisfactory yet.")
        image_height, image_width = disp.shape[:2]
        raster_settings = PointsRasterizationSettings(
            image_size=(image_height, image_width),
            radius = 30. / min(image_height, image_width),  # TODO: Check if it is good
            points_per_pixel = 1
        )
        rasterizer = PointsRasterizer(cameras=p3d_camera, raster_settings=raster_settings)
        pc_depth_map = rasterizer(Pointclouds(points=[pts], features=[pt_colors])).zbuf[0, ..., 0]
        mask = pc_depth_map > -1.

        true_points_depth = pc_depth_map[mask]
        depthmap_points_disp = disp[mask]
        weights = torch.ones_like(true_points_depth)
    else:
        _true_points_depth = p3d_camera.get_world_to_view_transform().transform_points(pts)[..., 2]
        _depthmap_points_disp, fov_mask = get_points_depth_in_depthmap(pts, disp, p3d_camera)

        true_points_depth = _true_points_depth[fov_mask] if use_fov_mask else _true_points_depth
        depthmap_points_disp = _depthmap_points_disp[fov_mask] if use_fov_mask else _depthmap_points_disp

        if image is not None and pt_colors is not None:
            raise NotImplementedError("Color weighting implementation is not satisfactory yet.")
            _image_points_colors = get_points_color_in_image(pts, image, p3d_camera)
            _true_points_colors = pt_colors

            image_points_colors = _image_points_colors[fov_mask]
            true_points_colors = _true_points_colors[fov_mask]

            weights = (image_points_colors * true_points_colors).sum(dim=-1)
        else:
            weights = torch.ones_like(true_points_depth)

    true_points_disp = 1. / true_points_depth

    beta_num = torch.sum(weights * true_points_disp * depthmap_points_disp) - torch.sum(weights * true_points_disp) * torch.sum(weights * depthmap_points_disp) / torch.sum(weights)
    beta_denom = torch.sum(weights * depthmap_points_disp ** 2) - torch.sum(weights * depthmap_points_disp) ** 2 / torch.sum(weights)
    beta = (beta_num / beta_denom).item()
    alpha = (torch.sum(weights * (true_points_disp - beta * depthmap_points_disp)) / torch.sum(weights)).item()

    if return_alpha_beta:
        return alpha, beta
    return 1. / (alpha + beta * disp)


def get_pointmap_from_sfm_data_with_depthanything(
    # SfM data
    pointmap_cameras:CamerasWrapper,
    training_cameras:CamerasWrapper,
    test_cameras:CamerasWrapper,
    sfm_xyz:torch.Tensor,
    sfm_col:torch.Tensor,
    image_sfm_points:list,
    image_dir:str,
    #
    n_images_in_pointmap:int=None,
    image_indices:list=None,
    randomize_images:bool=False,
    # DepthAnything V3
    depthanything_model_name:str='depth-anything/DA3NESTED-GIANT-LARGE',
    depth_model:DepthAnything3=None,
    # Misc
    device:torch.device='cuda',
):
    # Load / reuse DepthAnything V3 model
    if depth_model is None:
        model = load_model(
            model_name=depthanything_model_name,
            device=device,
        )
    else:
        model = depth_model

    # Get indices of images to process
    if randomize_images:
        raise NotImplementedError("Randomization of images is not implemented yet.")

    if image_indices is None and n_images_in_pointmap is None:
        raise ValueError("Either image_indices or n_images_in_pointmap must be provided.")
    if image_indices is not None:
        n_total_images = len(image_indices)
        n_images_in_pointmap = len(image_indices)
        frame_interval = 1
    else:
        n_total_images = len(pointmap_cameras)
        frame_interval = n_total_images // (n_images_in_pointmap - 1)

        image_indices = [_k * frame_interval for _k in range(n_images_in_pointmap)]
        n_images_in_pointmap = len(image_indices)

        n_total_images = len(image_indices)
        frame_interval = 1

    # Build dictionary for PointMap
    img_paths = []
    images = []
    original_images = []
    focals = []
    poses = []
    points3d = []
    confidence = []
    masks = []

    for cam_idx in image_indices:
        print(f"Processing frame {cam_idx}...")

        # Image path
        image_name = pointmap_cameras.gs_cameras[cam_idx].image_name.split('.')[0]
        img_path_i = os.path.join(
            image_dir,
            image_name
            + f".{os.listdir(image_dir)[0].split('.')[-1]}"
        )

        # Image
        gt_image = pointmap_cameras.gs_cameras[cam_idx].original_image.permute(1, 2, 0)
        _height, _width = gt_image.shape[:2]
        image_i = gt_image.view(_height, _width, 3)

        # Original image
        original_gt_image = training_cameras.gs_cameras[cam_idx].original_image.permute(1, 2, 0)
        height, width = original_gt_image.shape[:2]
        original_image_i = original_gt_image.view(height, width, 3)

        # Focal
        fx = fov2focal(pointmap_cameras.gs_cameras[cam_idx].FoVx, _width)
        fy = fov2focal(pointmap_cameras.gs_cameras[cam_idx].FoVy, _height)
        focal_i = torch.tensor([fx, fy], device=device)

        # Poses
        R = pointmap_cameras.gs_cameras[cam_idx].R
        T = pointmap_cameras.gs_cameras[cam_idx].T
        Rt = torch.cat([R.transpose(-1, -2), T.view(3, 1)], dim=1)
        Rt = torch.cat([Rt, torch.tensor([[0., 0., 0., 1.]], device=device)], dim=0)
        C2W = torch.linalg.inv(Rt)
        pose_i = C2W.view(4, 4)

        # Points3d
        cam_sfm_pts = sfm_xyz[image_sfm_points[image_name]]
        inv_depth = apply_depthanything(model, gt_image)# Compute inverse depths with DepthAnything V3
        # INSERT_YOUR_CODE
        # Resize inv_depth to 512 x 288 using bilinear interpolation
        inv_depth = torch.nn.functional.interpolate(
            inv_depth.unsqueeze(0).unsqueeze(0), size=(288, 512), mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)
        print("----------------------depth anything v3 value--------------------")
        print(inv_depth.min(), inv_depth.max())
        depth = fit_depth_to_point_cloud(
            disp=inv_depth,
            pts=cam_sfm_pts,
            training_cameras=pointmap_cameras,
            camera_idx=cam_idx,
            image=None,
            pt_colors=None,
            return_alpha_beta=False,
            use_rasterizer=False,
        )  # Scale depths to match the corresponding visible SfM points
        print("----------------------depth anything v3 value--------------------")
        print(depth.min(), depth.max())
        points3d_i = pointmap_cameras.backproject_depth(cam_idx=cam_idx, depth=depth).view(_height, _width, 3)
        print(points3d_i.max(), points3d_i.min())
        # Confidence
        confidence_i = torch.ones_like(points3d_i[..., 2]) * 1e8

        # Mask
        mask_i = torch.ones_like(points3d_i[..., 2], dtype=torch.bool)

        # Fill the data
        img_paths.append(img_path_i)
        images.append(image_i)
        original_images.append(original_image_i)
        focals.append(focal_i)
        poses.append(pose_i)
        points3d.append(points3d_i)
        confidence.append(confidence_i)
        masks.append(mask_i)

    pointmap = PointMapDepthAnything(
        scene_cameras=training_cameras,
        scene_eval_cameras=test_cameras,
        img_paths=img_paths,
        images=images,
        original_images=original_images,
        focals=focals,
        poses=poses,
        points3d=points3d,
        confidence=confidence,
        masks=masks,
        device=device,
    )

    return pointmap


def get_pointmap_from_colmap_scene_with_depthanything(
    # Scene data
    colmap_source_path,
    n_images_in_pointmap=None,
    image_indices=None,
    white_background=False,
    eval_split=False,
    eval_split_interval=8,
    max_img_size=1600,
    pointmap_img_size=512,
    randomize_images=False,
    # DepthAnything V3
    depthanything_model_name:str='depth-anything/DA3NESTED-GIANT-LARGE',
    # Misc
    device='cuda',
    return_sfm_data=False,
):
    # Load COLMAP scene data
    colmap_scene_dict = load_colmap_scene(
        colmap_source_path,
        load_gt_images=True,
        white_background=white_background,
        max_img_size=max_img_size,
        eval_split=eval_split,
        eval_split_interval=eval_split_interval,
        device=device,
    )
    training_cameras = colmap_scene_dict['training_cameras']
    test_cameras = colmap_scene_dict['test_cameras']
    sfm_xyz = colmap_scene_dict['sfm_xyz']
    sfm_col = colmap_scene_dict['sfm_col']
    image_sfm_points = colmap_scene_dict['image_sfm_points']

    colmap_scene_dict_pointmap = load_colmap_scene(  # Very inefficient, should change this to avoid reloading everything
        colmap_source_path,
        load_gt_images=True,
        white_background=white_background,
        max_img_size=pointmap_img_size,
        eval_split=eval_split,
        eval_split_interval=eval_split_interval,
        device=device,
    )
    pointmap_cameras = colmap_scene_dict_pointmap['training_cameras']

    image_dir = os.path.join(colmap_source_path, 'images')

    scene_pm = get_pointmap_from_sfm_data_with_depthanything(
        pointmap_cameras=pointmap_cameras,
        training_cameras=training_cameras,
        test_cameras=test_cameras,
        sfm_xyz=sfm_xyz,
        sfm_col=sfm_col,
        image_sfm_points=image_sfm_points,
        image_dir=image_dir,
        #
        n_images_in_pointmap=n_images_in_pointmap,
        image_indices=image_indices,
        randomize_images=randomize_images,
        # DepthAnything V3
        depthanything_model_name=depthanything_model_name,
        # Misc
        device=device,
    )
    if return_sfm_data:
        return scene_pm, colmap_scene_dict
    else:
        return scene_pm


def get_pointmap_from_dust3r_scene_with_depthanything(
    # Data
    scene_source_path,
    n_images_in_pointmap,
    image_indices=None,
    white_background=False,
    eval_split=False,
    eval_split_interval=8,
    max_img_size=1600,
    pointmap_img_size=512,
    randomize_images=False,
    # Output
    max_sfm_points=200_000,
    sfm_confidence_threshold=3.,
    average_focal_distances=False,
    # DUSt3R
    dust3r_checkpoint_path="./dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    dust3r_global_aligner_mode="pc",
    dust3r_global_aligner_niter=300,
    dust3r_global_aligner_schedule='cosine',
    dust3r_global_aligner_lr=0.01,
    prefilter_images=None,
    symmetrize_pairs=True,
    dust3r_batch_size=1,
    # DepthAnything V3
    depthanything_model_name:str='depth-anything/DA3NESTED-GIANT-LARGE',
    # Misc
    device='cuda',
    return_sfm_data=False,
):
    # Get SfM data from DUSt3R
    dust3r_sfm_data = compute_dust3r_scene(
        scene_source_path=scene_source_path,
        n_images_in_pointmap=n_images_in_pointmap,
        image_indices=image_indices,
        white_background=white_background,
        eval_split=eval_split,
        eval_split_interval=eval_split_interval,
        max_img_size=max_img_size,
        pointmap_img_size=pointmap_img_size,
        randomize_images=randomize_images,
        max_sfm_points=max_sfm_points,
        sfm_confidence_threshold=sfm_confidence_threshold,
        average_focal_distances=average_focal_distances,
        dust3r_checkpoint_path=dust3r_checkpoint_path,
        dust3r_global_aligner_mode=dust3r_global_aligner_mode,
        dust3r_global_aligner_niter=dust3r_global_aligner_niter,
        dust3r_global_aligner_schedule=dust3r_global_aligner_schedule,
        dust3r_global_aligner_lr=dust3r_global_aligner_lr,
        prefilter_images=prefilter_images,
        symmetrize_pairs=symmetrize_pairs,
        dust3r_batch_size=dust3r_batch_size,
        device=device,
    )

    # Check if images are in a subdirectory (common in COLMAP scenes)
    dir_content = os.listdir(scene_source_path)
    use_subdir = True
    for content in dir_content:
        if (
            content.endswith('.jpg')
            or content.endswith('.png')
            or content.endswith('.jpeg')
            or content.endswith('.JPG')
            or content.endswith('.PNG')
            or content.endswith('.JPEG')
        ):
            use_subdir = False
            print(f"Images found in {scene_source_path}. Using them for PointMap generation.")
    if use_subdir:
        source_path = os.path.join(scene_source_path, "images")
    else:
        source_path = scene_source_path

    # Get PointMap from SfM data with DepthAnything V3
    scene_pm = get_pointmap_from_sfm_data_with_depthanything(
        # SfM data
        pointmap_cameras=dust3r_sfm_data['pointmap_cameras'],
        training_cameras=dust3r_sfm_data['training_cameras'],
        test_cameras=dust3r_sfm_data['test_cameras'],
        sfm_xyz=dust3r_sfm_data['sfm_xyz'],
        sfm_col=dust3r_sfm_data['sfm_col'],
        image_sfm_points=dust3r_sfm_data['image_sfm_points'],
        image_dir=source_path,
        #
        n_images_in_pointmap=n_images_in_pointmap,
        image_indices=image_indices,
        randomize_images=randomize_images,
        # DepthAnything V3
        depthanything_model_name=depthanything_model_name,
        # Misc
        device=device,
    )
    if return_sfm_data:
        return scene_pm, dust3r_sfm_data
    else:
        return scene_pm


def get_pointmap_from_mast3r_scene_with_depthanything(
    # Data
    scene_source_path,
    n_images_in_pointmap,
    image_indices=None,
    white_background=False,
    eval_split=False,
    eval_split_interval=8,
    max_img_size=1600,
    pointmap_img_size=512,
    randomize_images=False,
    # Output
    max_sfm_points=200_000,
    sfm_confidence_threshold=0.,
    average_focal_distances=False,
    # MASt3R
    mast3r_scene_source_path=None,
    # DepthAnything V3
    depthanything_model_name:str='depth-anything/DA3NESTED-GIANT-LARGE',
    depth_model:DepthAnything3=None,
    # Misc
    device='cuda',
    return_sfm_data=False,
    return_mast3r_pointmap=False,
):
    # Get SfM data from MASt3R
    mast3r_sfm_data = compute_mast3r_scene(
        mast3r_scene_source_path=mast3r_scene_source_path,
        n_images_in_pointmap=n_images_in_pointmap,
        image_indices=image_indices,
        white_background=white_background,
        eval_split=eval_split,
        eval_split_interval=eval_split_interval,
        max_img_size=max_img_size,
        pointmap_img_size=pointmap_img_size,
        randomize_images=randomize_images,
        max_sfm_points=max_sfm_points,
        sfm_confidence_threshold=sfm_confidence_threshold,
        average_focal_distances=average_focal_distances,
        device=device,
        return_pointmap=return_mast3r_pointmap
    )
    if return_mast3r_pointmap:
        mast3r_sfm_data, mast3r_pm = mast3r_sfm_data

    # Check if images are in a subdirectory (common in COLMAP scenes)
    dir_content = os.listdir(scene_source_path)
    use_subdir = True
    for content in dir_content:
        if (
            content.endswith('.jpg')
            or content.endswith('.png')
            or content.endswith('.jpeg')
            or content.endswith('.JPG')
            or content.endswith('.PNG')
            or content.endswith('.JPEG')
        ):
            use_subdir = False
            print(f"Images found in {scene_source_path}. Using them for PointMap generation.")
    if use_subdir:
        source_path = os.path.join(scene_source_path, "images")
    else:
        source_path = scene_source_path

    # Get PointMap from SfM data with DepthAnything V3
    if (n_images_in_pointmap is None) and (image_indices is None):
        n_images_in_pointmap = len(mast3r_sfm_data['pointmap_cameras'])
    scene_pm = get_pointmap_from_sfm_data_with_depthanything(
        # SfM data
        pointmap_cameras=mast3r_sfm_data['pointmap_cameras'],
        training_cameras=mast3r_sfm_data['training_cameras'],
        test_cameras=mast3r_sfm_data['test_cameras'],
        sfm_xyz=mast3r_sfm_data['sfm_xyz'],
        sfm_col=mast3r_sfm_data['sfm_col'],
        image_sfm_points=mast3r_sfm_data['image_sfm_points'],
        image_dir=source_path,
        #
        n_images_in_pointmap=n_images_in_pointmap,
        image_indices=image_indices,
        randomize_images=randomize_images,
        # DepthAnything V3
        depthanything_model_name=depthanything_model_name,
        depth_model=depth_model,
        # Misc
        device=device,
    )

    export_pointmap_to_pcd(scene_pm, "aligned_pointmap.ply")
    if return_sfm_data:
        if return_mast3r_pointmap:
            return scene_pm, mast3r_sfm_data, mast3r_pm
        else:
            return scene_pm, mast3r_sfm_data
    else:
        return scene_pm
