import numpy as np
import rerun as rr
import json
import argparse
from pathlib import Path
import torch
import math
import os
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def focal2fov(focal, pixels):
    """Convert focal length to field of view."""
    if isinstance(focal, torch.Tensor) or isinstance(pixels, torch.Tensor):
        return 2 * torch.atan(pixels / (2 * focal))
    else:
        return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    """Convert field of view to focal length."""
    if isinstance(fov, torch.Tensor) or isinstance(pixels, torch.Tensor):
        return pixels / (2 * torch.tan(fov / 2))
    else:
        return pixels / (2 * math.tan(fov / 2))


def discover_timesteps(depth_base, camera_base, pattern="timestep_*"):
    """Discover all timestep directories in both depth and camera bases."""
    depth_path = Path(depth_base)
    camera_path = Path(camera_base)
    
    # Find timestep directories in depth base
    depth_timesteps = sorted(depth_path.glob(pattern))
    camera_timesteps = sorted(camera_path.glob(pattern))
    
    # Extract timestep indices
    depth_indices = set()
    for d in depth_timesteps:
        try:
            idx = int(d.name.split('_')[1])
            depth_indices.add(idx)
        except (IndexError, ValueError):
            continue
    
    camera_indices = set()
    for d in camera_timesteps:
        try:
            idx = int(d.name.split('_')[1])
            camera_indices.add(idx)
        except (IndexError, ValueError):
            continue
    
    # Find common timesteps
    common_indices = sorted(depth_indices & camera_indices)
    
    if not common_indices:
        raise ValueError(f"No common timesteps found between {depth_base} and {camera_base}")
    
    # Return list of (index, depth_path, camera_path) tuples
    timesteps = []
    for idx in common_indices:
        depth_dir = depth_path / f"timestep_{idx:06d}"
        camera_dir = camera_path / f"timestep_{idx:06d}"
        
        depth_file = depth_dir / "charts_data.npz"
        camera_file = camera_dir / "cameras.json"
        
        if depth_file.exists() and camera_file.exists():
            timesteps.append((idx, str(depth_file), str(camera_file)))
        else:
            print(f"Warning: Skipping timestep {idx} - missing files")
    
    return timesteps


def process_timestep(depth_path, camera_path, timestep_idx=None, color_mode='camera', 
                    confidence_colormap='coolwarm', confidence_range=None, use_log_confidence=True):
    """Process a single timestep and log to Rerun.
    
    Args:
        depth_path: Path to npz file with depth data
        camera_path: Path to JSON file with camera parameters
        timestep_idx: Timestep index for multi-timestep mode
        color_mode: 'camera' or 'confidence' for coloring strategy
        confidence_colormap: Matplotlib colormap name for confidence visualization
        confidence_range: Optional (min, max) tuple for confidence normalization
        use_log_confidence: If True, use log(C-1) instead of raw C values
    """
    
    # Load data
    print(f"Loading depth data from {depth_path}")
    depth_data = np.load(depth_path)
    prior_depths = depth_data['prior_depths']  # Shape: (4, 288, 512)
    depths = depth_data['depths']  # Shape: (4, 288, 512)
    scale_factor = depth_data['scale_factor'].item()  # Scalar value
    
    # Load confidence data if using confidence coloring
    if color_mode == 'confidence':
        confs = depth_data['confs']  # Shape: (4, 288, 512)
        
        # Transform to log-confidence if requested
        if use_log_confidence:
            # C = 1 + exp(C_hat), so C_hat = log(C - 1)
            # Use small epsilon to avoid log(0)
            log_confs = np.log(np.maximum(confs - 1.0, 1e-6))
            print(f"Log-confidence range: [{np.min(log_confs):.4f}, {np.max(log_confs):.4f}]")
            print(f"Log-confidence mean: {np.mean(log_confs):.4f}, std: {np.std(log_confs):.4f}")
        else:
            log_confs = confs
            print(f"Raw confidence range: [{np.min(confs):.4f}, {np.max(confs):.4f}]")
            print(f"Raw confidence mean: {np.mean(confs):.4f}, std: {np.std(confs):.4f}")
    
    print(f"Loading camera parameters from {camera_path}")
    with open(camera_path, 'r') as f:
        camera_data = json.load(f)
    
    focals = camera_data['focals']
    cams2world = camera_data['cams2world']
    
    # Get camera names from filepaths if available
    if 'filepaths' in camera_data:
        # Extract camera names from filepaths (e.g., './images/cam01_frame_000000.jpg' -> 'cam01')
        camera_names = []
        for filepath in camera_data['filepaths']:
            basename = os.path.basename(filepath)
            # Try to extract camera name (assuming format: cameraname_frame_XXXXXX.jpg)
            parts = basename.split('_')
            if len(parts) >= 3 and parts[1] == 'frame':
                camera_names.append(parts[0])
            else:
                # Fallback to generic naming
                camera_names.append(f'cam{len(camera_names)+1:02d}')
    else:
        # Fallback to generic camera names
        n_cameras = len(focals)
        camera_names = [f'cam{i+1:02d}' for i in range(n_cameras)]
    
    # Image dimensions
    h, w = 288, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert camera parameters to PyTorch3D format
    n_cameras = len(focals)
    
    # The input cams2world are camera-to-world matrices in OpenCV format
    # Simulate the GS camera creation process
    
    # Step 1: Convert focal lengths to FoV (as done in create_gs_cameras_from_pointmap)
    focals_tensor = torch.tensor(focals, dtype=torch.float32, device=device)
    FoVx = focal2fov(focals_tensor, w)
    FoVy = focal2fov(focals_tensor, h)
    
    # Step 2: Create GS cameras format (R, T)
    cams2world_tensor = torch.tensor(cams2world, dtype=torch.float32, device=device)
    world2cam = torch.inverse(cams2world_tensor)
    R_gs = world2cam[:, :3, :3]
    T_gs = world2cam[:, :3, 3] * scale_factor
    
    # Now follow the exact logic from convert_camera_from_gs_to_pytorch3d
    # Step 1: Reconstruct w2c from R and T
    w2c = torch.zeros(n_cameras, 4, 4).to(device)
    w2c[:, :3, :3] = R_gs
    w2c[:, :3, 3] = T_gs
    w2c[:, 3, 3] = 1
    
    # Step 2: Compute c2w and apply coordinate flip
    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1  # Flip y and z axes
    c2w = c2w[:, :3, :]
    
    # Step 3: Convert back to world2cam for PyTorch3D
    line = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device).expand(n_cameras, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2)
    T = T.squeeze(2)[:, :3]
    
    # Step 4: Apply PyTorch3D coordinate transformation
    pytorch3d_transform = torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float32, device=device)
    R_pytorch3d = R * pytorch3d_transform
    T_pytorch3d = T * pytorch3d_transform
    
    # Create intrinsics following the exact logic from convert_camera_from_gs_to_pytorch3d
    # Convert FoV back to focal (as done in line 564-565 of reference)
    image_height = torch.tensor([h] * n_cameras, dtype=torch.int, device=device)
    image_width = torch.tensor([w] * n_cameras, dtype=torch.int, device=device)
    fx = fov2focal(FoVx, image_width)
    fy = fov2focal(FoVy, image_height)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # PyTorch3D-compatible camera matrices
    # Intrinsics
    image_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    
    # Principal point in PyTorch3D convention (line 606 in reference)
    p0_pytorch3d = -(torch.cat([cx.view(-1, 1), cy.view(-1, 1)], dim=-1) - c0) / scale
    
    # Focal length in PyTorch3D convention (line 611 in reference)
    focal_pytorch3d = torch.cat([fx.view(-1, 1), fy.view(-1, 1)], dim=-1) / scale
    
    # Create calibration matrix (line 615-616 in reference)
    K = _get_sfm_calibration_matrix(
        n_cameras, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    
    # # Create PyTorch3D cameras
    # cameras = FoVPerspectiveCameras(
    #     R=R_pytorch3d,
    #     T=T_pytorch3d,
    #     K=K,
    #     device=device
    # )
    
    # Define colors for each camera (RGB values 0-255)
    # Generate colors automatically based on number of cameras
    def generate_camera_colors(n):
        """Generate distinct colors for n cameras."""
        if n <= 4:
            # Use predefined colors for up to 4 cameras
            colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 255, 0]   # Yellow
            ]
            return colors[:n]
        else:
            # Generate colors using HSV color space for more cameras
            import colorsys
            colors = []
            for i in range(n):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                colors.append([int(c * 255) for c in rgb])
            return colors
    
    camera_colors = generate_camera_colors(n_cameras)
    print(f"Processing {n_cameras} cameras: {', '.join(camera_names)}")
    
    # Setup confidence colormap if needed
    if color_mode == 'confidence':
        # Get colormap
        cmap = cm.get_cmap(confidence_colormap)
        
        # Determine normalization range
        if confidence_range is not None:
            vmin, vmax = confidence_range
        else:
            # Auto-determine range from all data
            vmin = np.min(log_confs)
            vmax = np.max(log_confs)
        
        # Create normalizer
        norm = Normalize(vmin=vmin, vmax=vmax)
        print(f"Using confidence normalization range: [{vmin:.4f}, {vmax:.4f}]")
        print(f"Using colormap: {confidence_colormap}")
    
    # Process each camera
    for i in range(n_cameras):
        print(f"Processing camera {i+1}/{n_cameras} ({camera_names[i]})...")
        
        # Get depth maps for this camera
        prior_depth = torch.tensor(prior_depths[i], dtype=torch.float32, device=device)
        aligned_depth = torch.tensor(depths[i], dtype=torch.float32, device=device)
        
        # Create pixel grid in NDC coordinates
        # PyTorch3D expects NDC coordinates where:
        # x: 1 (left) to -1 (right)
        # y: 1 (top) to -1 (bottom)
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Convert pixel coordinates to NDC
        xx_ndc = 1.0 - 2.0 * xx / (w - 1)
        yy_ndc = 1.0 - 2.0 * yy / (h - 1)
        xx_ndc = xx_ndc * w / h 
        
        # Stack coordinates (x_ndc, y_ndc, depth)
        xy_depth_prior = torch.stack([xx_ndc, yy_ndc, prior_depth], dim=-1).reshape(-1, 3)
        xy_depth_aligned = torch.stack([xx_ndc, yy_ndc, aligned_depth], dim=-1).reshape(-1, 3)
        
        # Filter valid depths
        valid_prior = (xy_depth_prior[:, 2] > 0) & (xy_depth_prior[:, 2] < 100)
        valid_aligned = (xy_depth_aligned[:, 2] > 0) & (xy_depth_aligned[:, 2] < 100)
        
        xy_depth_prior = xy_depth_prior[valid_prior]
        xy_depth_aligned = xy_depth_aligned[valid_aligned]
        
        # Unproject using PyTorch3D (single camera at a time)
        # Create single camera
        single_camera = FoVPerspectiveCameras(
            R=R_pytorch3d[i:i+1],
            T=T_pytorch3d[i:i+1],
            K=K[i:i+1],
            znear=0.0001,
            device=device
        )
        
        # Unproject points
        prior_points_world = single_camera.unproject_points(
            xy_depth_prior.unsqueeze(0),
            world_coordinates=True,
            from_ndc=True,
        ).squeeze(0)
        
        aligned_points_world = single_camera.unproject_points(
            xy_depth_aligned.unsqueeze(0),
            world_coordinates=True,
            from_ndc=True,
        ).squeeze(0)
        
        # Convert to numpy
        prior_points_np = prior_points_world.cpu().numpy()
        aligned_points_np = aligned_points_world.cpu().numpy()
        
        # Create color arrays
        if color_mode == 'confidence':
            # Get confidence values for this camera
            if use_log_confidence:
                camera_conf = log_confs[i]
            else:
                camera_conf = confs[i]
            
            # Flatten confidence and extract valid points
            flat_conf = camera_conf.flatten()
            
            # Get valid indices in the flattened array
            flat_prior_depth = prior_depths[i].flatten()
            flat_aligned_depth = depths[i].flatten()
            valid_prior_flat = (flat_prior_depth > 0) & (flat_prior_depth < 100)
            valid_aligned_flat = (flat_aligned_depth > 0) & (flat_aligned_depth < 100)
            
            # Extract confidence values for valid points
            prior_conf_values = flat_conf[valid_prior_flat]
            aligned_conf_values = flat_conf[valid_aligned_flat]
            
            # Apply colormap
            prior_colors_rgba = cmap(norm(prior_conf_values))
            aligned_colors_rgba = cmap(norm(aligned_conf_values))
            
            # Convert to RGB (0-255) and drop alpha channel
            prior_color = (prior_colors_rgba[:, :3] * 255).astype(np.uint8)
            aligned_color = (aligned_colors_rgba[:, :3] * 255).astype(np.uint8)
            
            # Log statistics for this camera
            print(f"  Camera {camera_names[i]} confidence stats:")
            print(f"    Prior points: min={prior_conf_values.min():.4f}, max={prior_conf_values.max():.4f}, mean={prior_conf_values.mean():.4f}")
            print(f"    Aligned points: min={aligned_conf_values.min():.4f}, max={aligned_conf_values.max():.4f}, mean={aligned_conf_values.mean():.4f}")
        else:
            # Use camera-based colors
            prior_color = np.tile(camera_colors[i], (len(prior_points_np), 1))
            aligned_color = np.tile(camera_colors[i], (len(aligned_points_np), 1))
        
        # Log individual camera point clouds for separate control
        print(f"Logging {len(prior_points_np)} prior depth points for {camera_names[i]}...")
        rr.log(f"/pointclouds/prior_depths/{camera_names[i]}", 
               rr.Points3D(prior_points_np, colors=prior_color, radii=0.01))
        
        print(f"Logging {len(aligned_points_np)} aligned depth points for {camera_names[i]}...")
        rr.log(f"/pointclouds/aligned_depths/{camera_names[i]}", 
               rr.Points3D(aligned_points_np, colors=aligned_color, radii=0.01))
        
        # Log camera using Rerun's Pinhole camera visualization (following Uni4D's approach)
        # Use the original camera poses directly, just like Uni4D does
        cam2world = cams2world_tensor[i].cpu().numpy()
        
        # Extract R and t
        world2cam = np.linalg.inv(cam2world)
        R_cam = world2cam[:3, :3]
        t_cam = world2cam[:3, 3] * scale_factor
        w2c = np.eye(4)
        w2c[:3, :3] = R_cam
        w2c[:3, 3] = t_cam
        w2c[3, 3] = 1.0
        cam2world = np.linalg.inv(w2c)
        R_cam = cam2world[:3, :3]
        t_cam = cam2world[:3, 3]

        # Rerun is quite strict about the rotation matrix being a valid SO3 matrix
        # It will not draw the frustum even if the determinant is just sligtly off due to numerical errors (e.g. 1.0001)
        if not np.isclose(np.linalg.det(R_cam), 1.0):
            print(f"Warning: Camera {i+1} has an invalid rotation matrix (det(R) != 1): {np.linalg.det(R_cam)}")
            # Re-normalize the rotation matrix - we should normalize rows for cam2world rotations
            R_cam = R_cam / np.linalg.norm(R_cam, axis=0, keepdims=True)

        # Also unproject depth to world coordinates using default cameras instead of Pytorch3D cameras
        x_cam = (xx - 511 / 2.0) * aligned_depth / fx[i]
        y_cam = (yy - 287 / 2.0) * aligned_depth / fy[i]
        z_cam = aligned_depth
        aligned_points_world_ = torch.stack([x_cam, y_cam, z_cam], dim=-1).reshape(
            -1, 3
        )
        aligned_points_world_ = torch.matmul(
            aligned_points_world_,
            torch.tensor(R_cam, dtype=torch.float32, device=device).T,
        ) + torch.tensor(t_cam, dtype=torch.float32, device=device)

        # Create intrinsic matrix K for Rerun visualization
        K_rerun = np.array([
            [focals[i], 0, w / 2.0],
            [0, focals[i], h / 2.0],
            [0, 0, 1]
        ])
        
        # Log camera with consistent naming
        rr.log(
            f"/cameras/{camera_names[i]}",
            rr.Pinhole(
                resolution=[w, h],
                image_from_camera=K_rerun,
            ),
            rr.Transform3D(translation=t_cam, mat3x3=R_cam),
        )


def main():
    parser = argparse.ArgumentParser(description='Visualize depth map comparison in 3D')
    
    # Single timestep mode (backward compatible)
    parser.add_argument('--data_path', type=str, 
                        default=None,
                        help='Path to single npz file (for single timestep mode)')
    parser.add_argument('--camera_path', type=str,
                        default=None,
                        help='Path to single camera JSON file (for single timestep mode)')
    
    # Multi-timestep mode
    parser.add_argument('--depth_base', type=str,
                        default='output/egohumans_legoassemble_new/depthanythingv2_aligned',
                        help='Base directory containing timestep folders with depth data')
    parser.add_argument('--camera_base', type=str,
                        default='output/egohumans_legoassemble_new/mast3r',
                        help='Base directory containing timestep folders with camera data')
    parser.add_argument('--max_timesteps', type=int, default=None,
                        help='Maximum number of timesteps to process (None for all)')
    parser.add_argument('--timestep_pattern', type=str, default='timestep_*',
                        help='Pattern to match timestep directories')
    
    # Output options
    parser.add_argument('--output', type=str, 
                        default='visualization_depth_comparison/depth_comparison.rrd',
                        help='Output path for the Rerun recording file')
    
    # Confidence visualization options
    parser.add_argument('--color_mode', type=str, choices=['camera', 'confidence'],
                        default='camera',
                        help='Color mode for point clouds: camera-based or confidence-based')
    parser.add_argument('--confidence_colormap', type=str, 
                        default='coolwarm',
                        help='Matplotlib colormap name for confidence visualization')
    parser.add_argument('--confidence_range', type=float, nargs=2,
                        default=None, metavar=('MIN', 'MAX'),
                        help='Manual confidence range for normalization (default: auto)')
    parser.add_argument('--use_log_confidence', action='store_true',
                        default=True,
                        help='Use log(C-1) instead of raw confidence values')
    parser.add_argument('--no_log_confidence', dest='use_log_confidence', 
                        action='store_false',
                        help='Use raw confidence values instead of log transformation')
    
    args = parser.parse_args()
    
    # Determine mode based on arguments
    single_timestep_mode = args.data_path is not None or args.camera_path is not None
    
    if single_timestep_mode:
        # Ensure both paths are provided for single timestep mode
        if args.data_path is None or args.camera_path is None:
            parser.error("Both --data_path and --camera_path must be provided for single timestep mode")
    else:
        # Use default example for single timestep if no args provided
        if args.data_path is None and args.camera_path is None and not os.path.exists(args.depth_base):
            args.data_path = 'output/egohumans_legoassemble_new/depthanythingv2_aligned/timestep_000000/charts_data.npz'
            args.camera_path = 'output/egohumans_legoassemble_new/mast3r/timestep_000000/cameras.json'
            single_timestep_mode = True
    
    # Initialize rerun
    rr.init("depth_comparison")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to file without spawning viewer
    rr.save(args.output)
    print(f"Saving recording to: {args.output}")
    
    if single_timestep_mode:
        # Process single timestep
        print("Running in single timestep mode")
        process_timestep(args.data_path, args.camera_path, 
                        color_mode=args.color_mode,
                        confidence_colormap=args.confidence_colormap,
                        confidence_range=args.confidence_range,
                        use_log_confidence=args.use_log_confidence)
    else:
        # Process multiple timesteps
        print("Running in multi-timestep mode")
        timesteps = discover_timesteps(args.depth_base, args.camera_base, args.timestep_pattern)
        
        if args.max_timesteps is not None:
            timesteps = timesteps[:args.max_timesteps]
        
        print(f"Found {len(timesteps)} timesteps to process")
        
        for timestep_idx, depth_path, camera_path in timesteps:
            print(f"\n=== Processing timestep {timestep_idx} ===")
            
            # Set Rerun time for this timestep
            rr.set_time_seconds("time", timestep_idx)
            
            # Process the timestep
            process_timestep(depth_path, camera_path, timestep_idx,
                            color_mode=args.color_mode,
                            confidence_colormap=args.confidence_colormap,
                            confidence_range=args.confidence_range,
                            use_log_confidence=args.use_log_confidence)
    
    print("\nVisualization complete!")
    print("Tips for multi-timestep viewing:")
    print("- Use the timeline at the bottom to scrub through timesteps")
    print("- Press space to play/pause the animation")
    print("- Use the entity tree to toggle visibility of different components")
    
    if args.color_mode == 'confidence':
        print("\nConfidence visualization notes:")
        if args.use_log_confidence:
            print("- Colors represent log-confidence values (Ĉ = log(C-1))")
            print("- Blue/cool colors = low confidence, Red/warm colors = high confidence")
        else:
            print("- Colors represent raw confidence values (C = 1 + exp(Ĉ))")
        print(f"- Using colormap: {args.confidence_colormap}")
        print("- Try different colormaps: viridis, plasma, turbo, coolwarm, RdBu")
    

if __name__ == "__main__":
    main()
