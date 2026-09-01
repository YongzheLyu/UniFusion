#!/usr/bin/env python3
"""
Evaluate depth against Aria MPS semidense point cloud projection.

Supports two modes:
- monofusion: depth files named by sequential data_idx (0, 1, 2, ...),
              mapped to test frames using "every 3 frames skip 1" rule
- deformable: depth files named by actual frame_idx, evaluate only on test frames
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Semidense point cloud loading
# ---------------------------------------------------------------------------

def load_semidense_points(
    path: str,
    threshold_invdep: float = 0.001,
    threshold_dep: float = 0.15,
) -> np.ndarray:
    """Load semidense 3D points, returning (N, 3) float32 array."""
    from projectaria_tools.core.mps import read_global_point_cloud

    points = read_global_point_cloud(path)
    before = len(points)

    points = [
        p for p in points
        if p.inverse_distance_std < threshold_invdep
        and p.distance_std < threshold_dep
    ]
    print(
        f"Quality filtering: {before} -> {len(points)} points "
        f"(inv_dist_std < {threshold_invdep}, dist_std < {threshold_dep})"
    )

    if len(points) == 0:
        raise ValueError(f"All {before} points filtered out")

    glb_pc = np.stack([p.position_world for p in points]).astype(np.float32)
    print(f"Loaded {len(glb_pc)} semidense points")
    return glb_pc


# ---------------------------------------------------------------------------
# Point cloud projection
# ---------------------------------------------------------------------------

def project_pc_to_depth(
    w2c: torch.Tensor,
    K: torch.Tensor,
    glb_pc: np.ndarray,
    img_wh: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Project point cloud onto camera view, returning (H, W) depth map."""
    w2c = w2c.to(device)
    K = K.to(device)

    pc = torch.from_numpy(glb_pc).float().to(device)
    ones = torch.ones((pc.shape[0], 1), device=device, dtype=pc.dtype)
    pc_hom = torch.cat([pc, ones], dim=1)

    c_pc_hom = pc_hom @ w2c.T
    c_pc = c_pc_hom[:, :3]
    x, y, z = c_pc[:, 0], c_pc[:, 1], c_pc[:, 2]

    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    coords = torch.stack([x, y, z], dim=0)
    uv = K @ coords
    u = uv[0, :] / uv[2, :]
    v = uv[1, :] / uv[2, :]

    u_pixel = torch.round(u).long()
    v_pixel = torch.round(v).long()

    W, H = img_wh
    in_bounds = (u_pixel >= 0) & (u_pixel < W) & (v_pixel >= 0) & (v_pixel < H)
    u_pixel = u_pixel[in_bounds]
    v_pixel = v_pixel[in_bounds]
    z = z[in_bounds]

    pixel_indices = v_pixel * W + u_pixel
    estimated_depth_flat = torch.full((H * W,), float("inf"), device=device, dtype=pc.dtype)

    if len(z) == 0:
        return estimated_depth_flat.view(H, W)

    sorted_indices = torch.argsort(pixel_indices)
    sorted_pixel_indices = pixel_indices[sorted_indices]
    sorted_z = z[sorted_indices]

    change_indices = torch.cat([
        torch.tensor([0], device=device),
        (sorted_pixel_indices[1:] != sorted_pixel_indices[:-1]).nonzero(as_tuple=True)[0] + 1,
        torch.tensor([len(sorted_pixel_indices)], device=device),
    ])

    for i in range(len(change_indices) - 1):
        start = change_indices[i].item()
        end = change_indices[i + 1].item()
        idx = sorted_pixel_indices[start].item()
        min_z = sorted_z[start:end].min()
        estimated_depth_flat[idx] = min_z

    return estimated_depth_flat.view(H, W)


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def load_cameras(cameras_json_path: str) -> dict:
    """Load cameras from pose_metadata.json or 3DGS cameras.json.

    Stored poses are c2w (camera-to-world), so convert them to w2c for
    projecting world-space semidense points into each camera. 3DGS
    cameras.json is a per-image list; for this static multiview evaluation
    we collapse it to one camera per camera id.
    """
    with open(cameras_json_path, "r") as f:
        data = json.load(f)

    cameras = {}

    if isinstance(data, list):
        seen_cam_ids = set()
        for cam in data:
            parts = cam["img_name"].split("_")
            if len(parts) < 3:
                continue
            cam_id = int(parts[1])
            if cam_id in seen_cam_ids:
                continue
            seen_cam_ids.add(cam_id)

            name = f"trajectory_undist_cam{cam_id:02d}"
            K = torch.tensor(
                [
                    [cam["fx"], 0.0, cam["width"] / 2.0],
                    [0.0, cam["fy"], cam["height"] / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
            c2w = torch.eye(4, dtype=torch.float32)
            c2w[:3, :3] = torch.tensor(cam["rotation"], dtype=torch.float32)
            c2w[:3, 3] = torch.tensor(cam["position"], dtype=torch.float32)
            cameras[name] = {
                "K": K,
                "w2c": torch.inverse(c2w),
                "img_wh": (cam["width"], cam["height"]),
            }
        return cameras

    cam_names = data["camera_names"]
    poses = data["poses"]
    intrinsics = data["intrinsics"]
    W = data.get("image_width", 512)
    H = data.get("image_height", 288)

    for i, name in enumerate(cam_names):
        K = torch.tensor(intrinsics[i], dtype=torch.float32)
        c2w = torch.tensor(poses[i], dtype=torch.float32)
        w2c = torch.inverse(c2w)
        cameras[name] = {"K": K, "w2c": w2c, "img_wh": (W, H)}

    return cameras


# ---------------------------------------------------------------------------
# Frame mapping logic
# ---------------------------------------------------------------------------

def build_test_frame_mapping(
    start: int,
    end: int,
    train_stride: int = 3,
) -> List[int]:
    """
    Build mapping from data_idx to actual frame_idx for test frames.

    Test frames = all frames except the train-frame sequence that starts at
    start and steps by train_stride.

    Example (train_stride=3, start=0):
        All frames:     0, 1, 2, 3, 4, 5, 6, 7, 8...
        Train frames:   0, 3, 6...
        Test frames:    1, 2, 4, 5, 7, 8...
        Mapping:        data_idx 0 -> 1, 1 -> 2, 2 -> 4, 3 -> 5...
    """
    all_frames = list(range(start, end + 1))
    train_frames = set(range(start, end + 1, train_stride))
    test_frames = [f for f in all_frames if f not in train_frames]
    return test_frames


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------

def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict:
    """Compute standard depth metrics on valid pixels.

    Returns both standard (all pixels) and inlier-only variants where
    only delta_3 inliers (max(p/g, g/p) < 1.25^3) are used, to reduce
    outlier impact on error metrics.
    """
    p = pred[valid_mask]
    g = gt[valid_mask]

    if len(p) == 0:
        return None

    thresh = torch.max(p / g, g / p)
    inlier = thresh < 1.25 ** 3

    # Inlier-only metrics (delta_3 pixels only)
    p_in = p[inlier]
    g_in = g[inlier]
    abs_rel_all = torch.abs(p - g) / g

    result = {
        # Standard (all pixels, Eigen et al.)
        "abs_rel": abs_rel_all.mean().item(),
        "abs_rel_p": (torch.abs(p - g) / p).mean().item(),
        "abs_rel_t95": abs_rel_all[abs_rel_all <= torch.quantile(abs_rel_all, 0.95)].mean().item(),
        "sq_rel": ((p - g) ** 2 / g).mean().item(),
        "rmse": ((p - g) ** 2).mean().sqrt().item(),
        "rmse_log": ((torch.log(p) - torch.log(g)) ** 2).mean().sqrt().item(),
        "delta_1": (thresh < 1.25).float().mean().item(),
        "delta_2": (thresh < 1.25 ** 2).float().mean().item(),
        "delta_3": (thresh < 1.25 ** 3).float().mean().item(),
        "n_pixels": int(valid_mask.sum().item()),
    }

    # Inlier-only metrics
    if len(p_in) > 0:
        result["inlier_abs_rel"] = (torch.abs(p_in - g_in) / g_in).mean().item()
        result["inlier_sq_rel"] = ((p_in - g_in) ** 2 / g_in).mean().item()
        result["inlier_rmse"] = ((p_in - g_in) ** 2).mean().sqrt().item()
        result["inlier_rmse_log"] = ((torch.log(p_in) - torch.log(g_in)) ** 2).mean().sqrt().item()
        result["inlier_n_pixels"] = int(inlier.sum().item())
    else:
        result["inlier_abs_rel"] = float("nan")
        result["inlier_sq_rel"] = float("nan")
        result["inlier_rmse"] = float("nan")
        result["inlier_rmse_log"] = float("nan")
        result["inlier_n_pixels"] = 0

    return result


# ---------------------------------------------------------------------------
# Mode-specific depth loading
# ---------------------------------------------------------------------------

def load_depth_monofusion(
    depth_dir: Path,
    cam_name: str,
    frame_idx: int,
):
    """
    Load depth for monofusion mode.

    Args:
        depth_dir: Root depth directory
        cam_name: Camera name (e.g., 'trajectory_undist_cam01')
        frame_idx: Actual frame index (e.g., 1, 2, 4, 5 for test frames)

    Returns:
        (depth_tensor, frame_idx) or (None, None) if not found

    Note: MonoFusion depth files use different naming convention.
    For camera 'trajectory_undist_cam01' and frame_idx=1:
        loads: {depth_dir}/camera_1/depth/0001_cam_Camera_1-depth.npy
    """
    import re

    # Extract camera number from name (e.g., 'trajectory_undist_cam01' -> '1')
    match = re.search(r'(\d+)$', cam_name)
    if not match:
        print(f"  Warning: Could not extract camera number from {cam_name}")
        return None, None

    cam_num = int(match.group(1))

    # MonoFusion format: {depth_dir}/camera_1/depth/0001_cam_Camera_1-depth.npy
    mono_cam_name = f"camera_{cam_num}"
    mono_file_cam_name = f"Camera_{cam_num}"

    cam_depth_dir = depth_dir / mono_cam_name / "depth"
    depth_path = cam_depth_dir / f"{frame_idx:04d}_cam_{mono_file_cam_name}-depth.npy"
    print(depth_path)
    if not depth_path.exists():
        return None, None

    depth = torch.from_numpy(np.load(str(depth_path))).float()
    return depth, frame_idx


def load_depth_deformable(
    depth_dir: Path,
    cam_name: str,
    data_idx: int,
    cam_names: List[str],
    frames_per_camera: int,
):
    """
    Load depth for deformable mode.

    Args:
        depth_dir: Root depth directory
        cam_name: Camera name
        data_idx: Sequential evaluated frame index for this camera (0, 1, 2, ...)
        cam_names: List of all camera names to determine camera index
        frames_per_camera: Number of frames per camera

    Returns:
        depth_tensor or None if not found

    Depth structure: all depth files are in one directory, named sequentially.
    For N cameras and M frames per camera:
        cam_names[0]: indices 0, 1, 2, ..., M-1
        cam_names[1]: indices M, M+1, ..., 2M-1
        etc.
    """
    # Determine camera index
    try:
        cam_idx = cam_names.index(cam_name)
    except ValueError:
        return None

    # Calculate the flat depth file index. Deformable depth files are packed by
    # camera block and sequential evaluated frame position, not by video frame id.
    depth_file_idx = cam_idx * frames_per_camera + data_idx

    depth_path = depth_dir / f"{depth_file_idx:05d}.npy"

    if depth_path.exists():
        return torch.from_numpy(np.load(str(depth_path))).float()

    return None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate depth against semidense point cloud",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True, choices=["monofusion", "deformable"],
        help="Evaluation mode: monofusion (sequential idx) or deformable (frame idx)"
    )
    parser.add_argument("--cameras-json", required=True, help="Path to pose_metadata.json")
    parser.add_argument("--semidense-points", required=True, help="Path to semidense_points.csv.gz")
    parser.add_argument("--depth-dir", required=True, help="Directory containing depth .npy files")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--mask-dir", default=None, help="Optional: directory with human masks")
    parser.add_argument("--device", default="cuda")

    # Frame range parameters
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end-frame", type=int, default=299, help="End frame index")
    parser.add_argument("--train-stride", type=int, default=3,
                        help="Training stride (test frames = non-multiples of this)")

    # Mode-specific parameters
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to evaluate (for monofusion: max data_idx)")
    parser.add_argument("--frames-per-camera", type=int, default=None,
                        help="Number of frames per camera (for deformable mode with flat depth structure)")

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "deformable" and args.frames_per_camera is None:
        parser.error("--frames-per-camera is required when using deformable mode")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    depth_dir = Path(args.depth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build test frame mapping
    test_frames = build_test_frame_mapping(args.start_frame, args.end_frame, args.train_stride)
    print(f"Test frames: {len(test_frames)} frames (from {test_frames[0]} to {test_frames[-1]})")

    # Load semidense points
    print(f"\nLoading semidense points from {args.semidense_points}")
    glb_pc = load_semidense_points(args.semidense_points)

    # Load cameras
    print(f"\nLoading cameras from {args.cameras_json}")
    cameras = load_cameras(args.cameras_json)
    print(f"  Found {len(cameras)} cameras: {list(cameras.keys())}")

    # Pre-project point cloud to each camera
    print("\nProjecting point cloud to cameras...")
    sparse_depths = {}
    for cam_name, cam in cameras.items():
        pc_depth = project_pc_to_depth(
            cam["w2c"], cam["K"], glb_pc, cam["img_wh"], device
        )
        n_valid = (pc_depth != float("inf")).sum().item()
        W, H = cam["img_wh"]
        coverage = n_valid / (H * W) * 100
        print(f"  {cam_name}: {n_valid} valid pixels ({coverage:.1f}% coverage)")
        sparse_depths[cam_name] = pc_depth

    # Evaluate
    all_metrics = {}
    all_flat_metrics = []

    for cam_name, cam in cameras.items():
        print(f"\nEvaluating camera: {cam_name}")
        pc_depth = sparse_depths[cam_name]
        W, H = cam["img_wh"]
        cam_metrics = []

        # Determine which frames to evaluate based on mode
        # For both modes, we evaluate on test frames (non-multiples of train_stride)
        frame_iterator = test_frames

        for data_idx, frame_idx in enumerate(frame_iterator):
            if args.mode == "monofusion":
                pred_depth, actual_frame = load_depth_monofusion(
                    depth_dir, cam_name, frame_idx
                )
                if pred_depth is None:
                    continue
                pred_depth = pred_depth.to(device)
            else:  # deformable
                # Get sorted camera names for indexing
                sorted_cam_names = sorted(cameras.keys())
                pred_depth = load_depth_deformable(
                    depth_dir, cam_name, data_idx, sorted_cam_names, args.frames_per_camera
                )
                if pred_depth is None:
                    continue
                pred_depth = pred_depth.to(device)
                actual_frame = frame_idx

            # Validate size
            if pred_depth.shape != (H, W):
                print(f"  Warning: Skipping frame {actual_frame} due to size mismatch "
                      f"(expected {(H, W)}, got {pred_depth.shape})")
                continue

            # Compute valid mask
            valid_mask = (pc_depth != float("inf")) & (pred_depth > 0)

            # Apply human mask if provided
            if args.mask_dir:
                mask_cam_name = cam_name
                mask_path = Path(args.mask_dir) / mask_cam_name / f"mask_{actual_frame:04d}.npy"
                if not mask_path.exists():
                    import re
                    match = re.search(r'(\d+)$', cam_name)
                    if match:
                        mask_cam_name = f"Camera_{int(match.group(1))}"
                        mask_path = Path(args.mask_dir) / mask_cam_name / f"mask_{actual_frame:04d}.npy"
                if mask_path.exists():
                    human_mask = np.load(str(mask_path))
                    if human_mask.ndim == 3:
                        human_mask = human_mask[0]
                    if human_mask.shape[:2] != (H, W):
                        import cv2
                        human_mask = cv2.resize(
                            human_mask, (W, H), interpolation=cv2.INTER_NEAREST
                        )
                    bg_mask = torch.from_numpy(human_mask == 0).to(device)
                    valid_mask = valid_mask & bg_mask
            #print(pred_depth)
            metrics = compute_depth_metrics(pred_depth, pc_depth, valid_mask)
            if metrics is None:
                continue

            metrics["frame_idx"] = frame_idx
            metrics["actual_frame"] = actual_frame
            cam_metrics.append(metrics)
            all_flat_metrics.append(metrics)

        all_metrics[cam_name] = cam_metrics
        print(f"  Evaluated {len(cam_metrics)} frames")

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)

    standard_keys = ["abs_rel", "abs_rel_p", "abs_rel_t95", "sq_rel", "rmse", "rmse_log", "delta_1", "delta_2", "delta_3"]
    inlier_error_keys = ["inlier_abs_rel", "inlier_sq_rel", "inlier_rmse", "inlier_rmse_log"]
    all_metric_keys = standard_keys + inlier_error_keys + ["inlier_n_pixels"]

    def _mean_values(metrics_list, key):
        values = [m[key] for m in metrics_list if not (isinstance(m[key], float) and np.isnan(m[key]))]
        return float(np.mean(values)) if values else float("nan")

    summary = {}
    for cam_name, metrics_list in all_metrics.items():
        if not metrics_list:
            continue
        cam_summary = {k: _mean_values(metrics_list, k) for k in all_metric_keys}
        cam_summary["n_frames"] = len(metrics_list)
        cam_summary["avg_n_pixels"] = float(np.mean([m["n_pixels"] for m in metrics_list]))
        summary[cam_name] = cam_summary

    # Overall
    if all_flat_metrics:
        overall = {k: _mean_values(all_flat_metrics, k) for k in all_metric_keys}
        overall["n_frames"] = len(all_flat_metrics)
        overall["avg_n_pixels"] = float(np.mean([m["n_pixels"] for m in all_flat_metrics]))
        summary["overall"] = overall

    # Print standard metrics table
    print("\n" + "=" * 80)
    print("STANDARD METRICS (Eigen et al.)")
    print("=" * 80)
    header = f"{'Camera':<24}" + "".join(f"{k:>12}" for k in standard_keys) + f"{'n_frames':>12}"
    print(header)
    print("-" * len(header))
    for cam_name in all_metrics:
        if cam_name not in summary:
            print(f"{cam_name:<12}  (no valid frames)")
            continue
        s = summary[cam_name]
        row = f"{cam_name:<24}" + "".join(f"{s[k]:>12.4f}" for k in standard_keys) + f"{int(s['n_frames']):>12d}"
        print(row)
    if "overall" in summary:
        s = summary["overall"]
        print("-" * len(header))
        row = f"{'Overall':<24}" + "".join(f"{s[k]:>12.4f}" for k in standard_keys) + f"{int(s['n_frames']):>12d}"
        print(row)
    print("=" * 80)

    # Print inlier metrics table
    inlier_display = ["abs_rel", "sq_rel", "rmse", "rmse_log"]
    print("\n" + "=" * 80)
    print("INLIER METRICS (delta_3 pixels only, max(p/g,g/p) < 1.25^3)")
    print("=" * 80)
    header = f"{'Camera':<12}" + "".join(f"{k:>10}" for k in inlier_display) + f"{'inlier_n':>12}"
    print(header)
    print("-" * len(header))
    for cam_name in all_metrics:
        if cam_name not in summary:
            print(f"{cam_name:<12}  (no valid frames)")
            continue
        s = summary[cam_name]
        row = f"{cam_name:<12}" + "".join(f"{s[k]:>10.4f}" for k in inlier_error_keys) + f"{int(s.get('inlier_n_pixels', 0)):>12d}"
        print(row)
    if "overall" in summary:
        s = summary["overall"]
        print("-" * len(header))
        row = f"{'Overall':<12}" + "".join(f"{s[k]:>10.4f}" for k in inlier_error_keys) + f"{int(s.get('inlier_n_pixels', 0)):>12d}"
        print(row)
    print("=" * 80)

    # Save results
    results = {
        "mode": args.mode,
        "cameras_json": args.cameras_json,
        "semidense_points": args.semidense_points,
        "depth_dir": str(depth_dir),
        "train_stride": args.train_stride,
        "test_frames": len(test_frames),
        "per_camera": summary,
    }

    results_path = output_dir / "depth_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
