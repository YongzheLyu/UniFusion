import torch
import numpy as np
from typing import Union


def compute_plucker_rays(
    camera,
    device: Union[str, torch.device] = 'cuda'
):
    """Compute Plucker ray coordinates for each pixel in the camera image.

    Plucker coordinates represent a 3D line using two 3D vectors:
    - Direction vector d (normalized ray direction)
    - Moment vector m = o × d (cross product of camera center and direction)

    Args:
        camera: A GSCamera object with the following attributes:
            - R: Rotation matrix (3, 3)
            - T: Translation vector (3,)
            - image_height: Image height in pixels
            - image_width: Image width in pixels
            - focal_x: Focal length in x direction
            - focal_y: Focal length in y direction
        device: Device to compute on. Defaults to 'cuda'.

    Returns:
        dict: Dictionary containing:
            - 'directions': (H, W, 3) - Normalized ray directions in world coordinates
            - 'moments': (H, W, 3) - Moment vectors (camera_center × direction) in world coordinates
            - 'plucker': (H, W, 6) - Concatenated Plucker coordinates [d, m]
            - 'camera_center': (3,) - Camera center in world coordinates
    """
    # Get camera parameters
    H = camera.image_height
    W = camera.image_width
    fx = camera.focal_x
    fy = camera.focal_y
    
    # Camera center in world coordinates
    # C = -R^T @ T (world coordinates)
    R = camera.R.to(device)
    T = camera.T.to(device)
    camera_center = -R.T @ T  # (3,)
    
    # Principal point (assumed at image center)
    cx = W / 2.0
    cy = H / 2.0
    
    # Create pixel coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # Convert pixel coordinates to normalized camera coordinates
    # x_norm = (x - cx) / fx
    # y_norm = (y - cy) / fy
    x_norm = (x_coords - cx) / fx  # (H, W)
    y_norm = (y_coords - cy) / fy  # (H, W)
    
    # Create direction vectors in camera coordinates (z=1 for normalized coordinates)
    # In camera coordinates, ray direction is [x_norm, y_norm, 1]
    directions_cam = torch.stack([
        x_norm,
        y_norm,
        torch.ones_like(x_norm)
    ], dim=-1)  # (H, W, 3)
    
    # Normalize directions in camera space
    directions_cam = directions_cam / torch.norm(directions_cam, dim=-1, keepdim=True)
    
    # Transform directions from camera coordinates to world coordinates
    # d_world = R @ d_cam
    directions_world = directions_cam @ R.T  # (H, W, 3)
    
    # Normalize again after transformation (should already be normalized, but for safety)
    directions_world = directions_world / torch.norm(directions_world, dim=-1, keepdim=True)
    
    # Compute moment vectors: m = camera_center × direction
    # Expand camera_center to (H, W, 3) for broadcasting
    camera_center_expanded = camera_center.unsqueeze(0).unsqueeze(0).expand(H, W, -1)  # (H, W, 3)
    
    # Cross product: m = o × d
    moments = torch.cross(camera_center_expanded, directions_world, dim=-1)  # (H, W, 3)
    
    # Concatenate to form Plucker coordinates [d, m]
    plucker = torch.cat([directions_world, moments], dim=-1)  # (H, W, 6)
    
    return {
        'directions': directions_world,  # (H, W, 3)
        'moments': moments,  # (H, W, 3)
        'plucker': plucker,  # (H, W, 6)
        'camera_center': camera_center,  # (3,)
    }


def compute_plucker_rays_batch(
    cameras,
    device: Union[str, torch.device] = 'cuda'
):
    """Compute Plucker ray coordinates for each pixel in multiple camera images.

    Args:
        cameras: Either a CamerasWrapper object or a list of GSCamera objects.
        device: Device to compute on. Defaults to 'cuda'.

    Returns:
        list: List of dictionaries, each containing the same keys as compute_plucker_rays.
    """
    if hasattr(cameras, 'gs_cameras'):
        # CamerasWrapper
        camera_list = cameras.gs_cameras
    elif isinstance(cameras, list):
        # List of GSCamera
        camera_list = cameras
    else:
        # Single camera
        return [compute_plucker_rays(cameras, device)]
    
    results = []
    for camera in camera_list:
        results.append(compute_plucker_rays(camera, device))
    
    return results

