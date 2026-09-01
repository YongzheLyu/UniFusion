import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from matcha.dm_scene.meshes import get_manifold_meshes_from_pointmaps
from matcha.dm_scene.cameras import CamerasWrapper, create_gs_cameras_from_pointmap, rescale_cameras


def compute_initial_depths(scene_pm, scale_factor=1.0):
    """Compute initial depth maps for each camera/chart in a scene pointmap.

    Args:
        scene_pm: PointMap object containing points3d, images
        scale_factor: Scale factor to apply to points

    Returns:
        initial_depths: torch.Tensor of shape (N, H, W) where N is number of charts
    """
    pm_h, pm_w = scene_pm.points3d.shape[1:3]

    # Create cameras from pointmap
    cam_list = create_gs_cameras_from_pointmap(
        scene_pm,
        image_resolution=1,
        load_gt_images=True,
        max_img_size=1600,
        use_original_image_size=True,
        average_focal_distances=False,
        verbose=False,
    )
    pointmap_cameras = CamerasWrapper(cam_list, no_p3d_cameras=False)

    # Scale points
    pt_maps = scale_factor * scene_pm.points3d
    imgs = scene_pm.images

    # Create mesh to extract vertices
    manifolds, _ = get_manifold_meshes_from_pointmaps(
        pt_maps, imgs, masks=None, return_single_mesh_object=True, return_manifold_idx=True
    )
    verts = manifolds.verts_packed().cuda()
    print(verts.device)
    # Compute initial depths
    initial_depths = torch.cat([
        pointmap_cameras. p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
            verts.reshape(scene_pm.points3d.shape)[i_chart].reshape(-1, 3)
        )[..., 2].reshape(1, pm_h, pm_w) for i_chart in range(len(pointmap_cameras))
    ], dim=0)  # (N, H, W)

    return initial_depths


def visualize_initial_depths(data, frame_idx, output_dir=None, max_cols=4):
    """Visualize initial depths and reference depths for a specific frame across all cameras.

    Args:
        data: Dictionary containing the loaded pkl data
        frame_idx: Frame index to visualize
        output_dir: Optional directory to save images
        max_cols: Maximum number of columns in the visualization grid
    """
    temporal_scene_pms = data['temporal_scene_pms']
    temporal_reference_data = data['temporal_reference_data']
    temporal_frame_indices = data['temporal_frame_indices']
    scale_factor = data.get('scale_factor', 1.0)

    # Find the timestamp index for the requested frame
    timestamp_idx = None
    for i, f_idx in enumerate(temporal_frame_indices):
        if f_idx == frame_idx:
            timestamp_idx = i
            break

    if timestamp_idx is None:
        raise ValueError(f"Frame {frame_idx} not found. Available frames: {temporal_frame_indices}")

    scene_pm = temporal_scene_pms[timestamp_idx]
    n_charts = len(scene_pm.images)
    pm_h, pm_w = scene_pm.points3d.shape[1:3]

    print(f"Visualizing frame {frame_idx} (timestamp index {timestamp_idx})")
    print(f"Number of cameras/charts: {n_charts}")
    print(f"Pointmap resolution: {pm_h} x {pm_w}")
    print(f"Scale factor: {scale_factor}")

    # Compute initial depths
    initial_depths = compute_initial_depths(scene_pm, scale_factor)
    print(f"Initial depths shape: {initial_depths.shape}")

    # Get reference depths for this timestamp
    reference_depths = temporal_reference_data[timestamp_idx]
    if isinstance(reference_depths, torch.Tensor):
        reference_depths = reference_depths.cpu()
    print(f"Reference depths shape: {reference_depths.shape}")

    # Analyze outliers in reference depths
    ref_depths_np = reference_depths.numpy()
    total_pixels = ref_depths_np.size
    valid_pixels = ref_depths_np[ref_depths_np > 0]
    valid_count = valid_pixels.size

    print(f"\nReference Depth Outlier Analysis:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Valid (non-zero) pixels: {valid_count} ({valid_count/total_pixels*100:.2f}%)")

    # Count pixels with depth > 100, > 1000, > 10000
    gt_10 =  np.sum(ref_depths_np > 10)
    gt_100 = np.sum(ref_depths_np > 100)
    gt_1000 = np.sum(ref_depths_np > 1000)
    gt_10000 = np.sum(ref_depths_np > 10000)
    gt_neg = np.sum(ref_depths_np < 0 )
    print(f"  Pixels with depth <0: {gt_neg} ({gt_neg/total_pixels*100:.4f}%)")
    print(f"  Pixels with depth > 10: {gt_10} ({gt_10/total_pixels*100:.4f}%)")
    print(f"  Pixels with depth > 100: {gt_100} ({gt_100/total_pixels*100:.4f}%)")
    print(f"  Pixels with depth > 1000: {gt_1000} ({gt_1000/total_pixels*100:.4f}%)")
    print(f"  Pixels with depth > 10000: {gt_10000} ({gt_10000/total_pixels*100:.4f}%)")

    # Calculate grid layout
    n_cols = min(n_charts, max_cols)
    n_rows = (n_charts + n_cols - 1) // n_cols

    # Create visualization with 2 rows per camera (initial and reference)
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(4 * n_cols, 8 * n_rows))
    if n_rows * 2 == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    elif n_rows * 2 == 1:
        axes = [axes]

    # Find global min/max for consistent color scale across both initial and reference
    depth_min = min(initial_depths.min().item(), reference_depths.min().item())
    depth_max = max(initial_depths.max().item(), reference_depths.max().item())
    print(f"Depth range: [{depth_min:.4f}, {depth_max:.4f}]")
    print("ref depth range:", reference_depths.min().item(), reference_depths.max().item())
    # Plot each camera's depth maps (both initial and reference)
    for i in range(n_charts):
        col = i % n_cols
        row_initial = (i // n_cols) * 2
        row_reference = row_initial + 1

        # Plot initial depth
        depth_map_initial = initial_depths[i].cpu().numpy()
        im_initial = axes[row_initial][col].imshow(depth_map_initial, cmap='plasma', vmin=depth_min, vmax=depth_max)
        axes[row_initial][col].set_title(f'Initial - Camera {i}')
        axes[row_initial][col].axis('off')
        plt.colorbar(im_initial, ax=axes[row_initial][col], fraction=0.046, pad=0.04)

        # Plot reference depth
        depth_map_reference = reference_depths[i].cpu().numpy()
        im_reference = axes[row_reference][col].imshow(depth_map_reference, cmap='plasma', vmin=depth_min, vmax=depth_max)
        axes[row_reference][col].set_title(f'Reference - Camera {i}')
        axes[row_reference][col].axis('off')
        plt.colorbar(im_reference, ax=axes[row_reference][col], fraction=0.046, pad=0.04)

    # Hide unused subplots
    total_rows = n_rows * 2
    for i in range(n_charts, n_rows * n_cols):
        for row_offset in range(2):
            row = (i // n_cols) * 2 + row_offset
            col = i % n_cols
            if row < total_rows:
                axes[row][col].axis('off')

    plt.suptitle(f'Initial vs Reference Depths - Frame {frame_idx} (Timestamp {timestamp_idx})', fontsize=16)
    plt.tight_layout()

    # Save or show
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f'frame_{frame_idx:05d}_depths_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    plt.show()

    # Also create individual visualizations for each camera (both initial and reference)
    print("\nCreating individual visualizations for each camera...")
    for i in range(n_charts):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot initial depth
        depth_map_initial = initial_depths[i].cpu().numpy()
        im_initial = axes[0].imshow(depth_map_initial, cmap='plasma', vmin=depth_min, vmax=depth_max)
        axes[0].set_title(f'Initial Depth - Frame {frame_idx}, Camera {i}')
        axes[0].axis('off')
        plt.colorbar(im_initial, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot reference depth
        depth_map_reference = reference_depths[i].cpu().numpy()
        print(depth_map_reference.shape)
        im_reference = axes[1].imshow(depth_map_reference, cmap='plasma', vmin=depth_min, vmax=depth_max)
        axes[1].set_title(f'Reference Depth - Frame {frame_idx}, Camera {i}')
        axes[1].axis('off')
        plt.colorbar(im_reference, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if output_dir:
            save_path = output_dir / f'frame_{frame_idx:05d}_camera_{i:02d}_depth_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        else:
            plt.show()

        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize initial depths from preprocessed pkl file')

    parser.add_argument('-p', '--pkl_file', type=str, required=True,
                       help='Path to preprocessed data file (.pkl)')
    parser.add_argument('-f', '--frame', type=int, required=True,
                       help='Frame index to visualize')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                       help='Output directory for saving images (optional)')
    parser.add_argument('--max_cols', type=int, default=4,
                       help='Maximum number of columns in visualization grid (default: 4)')

    args = parser.parse_args()

    print(f"Loading pkl file from: {args.pkl_file}")
    with open(args.pkl_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Available frames: {data['temporal_frame_indices']}")

    visualize_initial_depths(
        data=data,
        frame_idx=args.frame,
        output_dir=args.output_dir,
        max_cols=args.max_cols,
    )
