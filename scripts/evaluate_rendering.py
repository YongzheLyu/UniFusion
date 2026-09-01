#!/usr/bin/env python3
"""
Evaluation script for UniFusion rendering results.

Computes PSNR, SSIM, and LPIPS metrics between predicted rendered images and ground truth images.
Also computes foreground-only PSNR/SSIM/LPIPS using SAM masks.

Input formats:
1. UniFusion format: /path/to/test/ours_XXXXX/renders/ and /path/to/test/ours_XXXXX/gt/
2. Monofusion format: /path/to/monofusion_results/scene_name/cam_X/predicted/ and .../ground_truth/
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


def get_frame_number(filename: str) -> int:
    """Extract frame number from XXXXX.png pattern (e.g., 00000.png -> 0)."""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid filename pattern: {filename}")


def render_idx_to_frame(render_idx: int) -> int:
    """
    Convert render index to actual frame number.
    Training takes every 3rd frame (skips frame 3, 6, 9, ...).
    Formula: frame = render_idx + render_idx // 2 + 1

    Mapping:
    render_idx 0 -> frame 1
    render_idx 1 -> frame 2
    render_idx 2 -> frame 4 (skip 3)
    render_idx 3 -> frame 5
    render_idx 4 -> frame 7 (skip 6)
    ...
    """
    return render_idx + render_idx // 2 + 1


def render_idx_to_camera_and_frame(render_idx: int, frames_per_cam: int = 200) -> Tuple[int, int]:
    """
    Convert render index to (camera_index, actual_frame_number).

    Args:
        render_idx: The render index (0-based)
        frames_per_cam: Number of frames per camera (default 200)

    Returns:
        (camera_idx, actual_frame_number)

    Example (assuming 200 frames per cam, render_idx_to_frame mapping):
    render_idx 0 -> cam 0, frame 1
    render_idx 199 -> cam 0, frame 299
    render_idx 200 -> cam 1, frame 301 (skipped 300 since 300/3=100)
    """
    actual_frame = render_idx_to_frame(render_idx)
    camera_idx = render_idx // frames_per_cam
    return camera_idx, actual_frame


def load_mask(mask_dir: Path, camera_name: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load mask for a given camera and frame index.

    Mask structure:
    - mask_dir/{camera_name}/mask_{frame_idx:04d}.npy
    - npy file contains mask array with shape (H, W) or (1, H, W)

    Args:
        mask_dir: Root directory containing camera subdirectories
        camera_name: Camera directory name (e.g., 'Camera_1')
        frame_idx: Frame index (0-based)

    Returns:
        Mask array with shape (H, W), values in [0, 1], or None if not found
    """
    camera_mask_dir = mask_dir / camera_name
    if not camera_mask_dir.exists():
        return None

    # Load npy file
    npy_path = camera_mask_dir / f"mask_{frame_idx:04d}.npy"
    if not npy_path.exists():
        return None

    try:
        mask = np.load(str(npy_path))
        # Squeeze to (H, W) if needed
        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] == 1 else np.squeeze(mask)
        if mask.ndim == 2:
            return mask.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to load mask from {npy_path}: {e}")
    return None


def discover_camera_names(mask_dir: Path) -> List[str]:
    """
    Discover available camera names from mask directory.

    Args:
        mask_dir: Root directory containing camera subdirectories

    Returns:
        List of camera directory names (e.g., ['ball_undist_cam00', 'ball_undist_cam01', ...])
    """
    if not mask_dir.exists():
        return []

    camera_names = []
    for item in sorted(mask_dir.iterdir()):
        if item.is_dir():
            camera_names.append(item.name)

    return camera_names


def is_monofusion_format(input_dir: Path) -> bool:
    """
    Check if input directory follows monofusion format.
    Monofusion format: cam_X/predicted/ and cam_X/ground_truth/

    Args:
        input_dir: Root directory to check

    Returns:
        True if monofusion format is detected
    """
    if not input_dir.exists():
        return False

    # Check for cam_* directories with predicted/ and ground_truth/ subdirs
    for item in sorted(input_dir.iterdir()):
        if item.is_dir() and item.name.startswith("cam_"):
            pred_dir = item / "predicted"
            gt_dir = item / "ground_truth"
            if pred_dir.exists() and gt_dir.exists():
                return True
    return False


def load_monofusion_images(input_dir: Path) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[str]]:
    """
    Load images from monofusion format directory.

    Monofusion format:
    - input_dir/cam_X/predicted/pred_frame_XXXX.png
    - input_dir/cam_X/ground_truth/gt_frame_XXXX.png

    Frame numbers are assigned globally across all cameras (cam_1: 0-199, cam_2: 200-399, etc.)

    Args:
        input_dir: Root directory containing cam_X folders

    Returns:
        (rendered_dict, gt_dict, camera_names)
        rendered_dict: {frame_num: image_array}
        gt_dict: {frame_num: image_array}
        camera_names: List of camera directory names
    """
    rendered = {}
    gt = {}
    camera_names = []

    # Discover all cam_X directories
    for item in sorted(input_dir.iterdir()):
        if item.is_dir() and item.name.startswith("cam_"):
            camera_names.append(item.name)

    if not camera_names:
        raise ValueError(f"No cam_X directories found in {input_dir}")

    print(f"Found {len(camera_names)} cameras: {camera_names}")

    frames_per_cam = 200  # Assume 200 frames per camera

    for cam_idx, cam_name in enumerate(camera_names):
        cam_dir = input_dir / cam_name
        pred_dir = cam_dir / "predicted"
        gt_dir = cam_dir / "ground_truth"

        if not pred_dir.exists() or not gt_dir.exists():
            print(f"Warning: Missing predicted or ground_truth directory for {cam_name}")
            continue

        # Load predicted images
        pred_images = {}
        for img_path in sorted(pred_dir.glob("*.png")):
            frame_num = get_frame_number(img_path.name)
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_images[frame_num] = img.astype(np.float32) / 255.0

        # Load ground truth images
        gt_images = {}
        for img_path in sorted(gt_dir.glob("*.png")):
            frame_num = get_frame_number(img_path.name)
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt_images[frame_num] = img.astype(np.float32) / 255.0

        print(f"  {cam_name}: {len(pred_images)} predicted, {len(gt_images)} GT images")

        # Assign global frame numbers
        # cam_0: 0-199, cam_1: 200-399, etc.
        for local_frame in pred_images:
            print(round(local_frame*2/3) + local_frame%3 -1)
            global_frame = cam_idx * frames_per_cam + round(local_frame*2/3) + local_frame%3 -1  # local_frame is 1-indexed
            rendered[global_frame] = pred_images[local_frame]

        for local_frame in gt_images:
            global_frame = cam_idx * frames_per_cam + round(local_frame*2/3) + local_frame%3 -1
            gt[global_frame] = gt_images[local_frame]

    return rendered, gt, camera_names


def compute_fg_metrics(pairs, mask_cam_dir: Path, bbox_pad: float = 0.4, device="cpu"):
    """Compute foreground-only PSNR / SSIM / LPIPS on bbox crops.

    Masks are looked up as mask_cam_dir/mask_{frame_idx:04d}.npy.
    Returns dict with keys: fg_psnr, fg_ssim, fg_lpips, num_frames, per_frame.
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)

    per_frame = []
    for frame_idx, pred_np, gt_np in tqdm(pairs, desc="    foreground", leave=False):
        mask_path = mask_cam_dir / f"mask_{frame_idx:04d}.npy"
        if not mask_path.exists():
            continue

        mask = np.load(str(mask_path))
        if mask.ndim == 3:
            mask = mask[0]
        # Resize mask if needed
        if mask.shape[:2] != pred_np.shape[:2]:
            mask = cv2.resize(mask, (pred_np.shape[1], pred_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        H, W = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue

        # Padded bounding box
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        bh, bw = y1 - y0, x1 - x0
        pad_y, pad_x = int(bh * bbox_pad), int(bw * bbox_pad)
        y0, y1 = max(0, y0 - pad_y), min(H, y1 + pad_y)
        x0, x1 = max(0, x0 - pad_x), min(W, x1 + pad_x)

        pred_crop = pred_np[y0:y1, x0:x1].copy()
        gt_crop = gt_np[y0:y1, x0:x1].copy()
        mask_crop = mask[y0:y1, x0:x1]

        # White-fill background pixels so they contribute zero error
        bg = mask_crop == 0
        pred_crop[bg] = 1.0
        gt_crop[bg] = 1.0

        # PSNR from raw squared error (for global pooling)
        diff = (pred_crop - gt_crop).astype(np.float64)
        sq_error = float(np.sum(diff ** 2))
        total = diff.shape[0] * diff.shape[1] * 3  # H * W * 3
        mse = sq_error / total
        psnr_val = float("inf") if mse == 0 else -10.0 * np.log(mse) / np.log(10.0)

        # SSIM
        pred_t = torch.from_numpy(pred_crop).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_t = torch.from_numpy(gt_crop).permute(2, 0, 1).unsqueeze(0).to(device)
        ssim_val = ssim_metric(pred_t, gt_t).item()
        ssim_metric.reset()

        # LPIPS
        lpips_val = lpips_metric(pred_t, gt_t).item()
        lpips_metric.reset()

        per_frame.append({
            "frame_idx": frame_idx,
            "fg_psnr": psnr_val,
            "fg_ssim": ssim_val,
            "fg_lpips": lpips_val,
            "fg_sum_squared_error": sq_error,
            "fg_total": total,
        })

    psnr_vals = [f["fg_psnr"] for f in per_frame if np.isfinite(f["fg_psnr"])]
    ssim_vals = [f["fg_ssim"] for f in per_frame if np.isfinite(f["fg_ssim"])]
    lpips_vals = [f["fg_lpips"] for f in per_frame if np.isfinite(f["fg_lpips"])]

    return {
        "fg_psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "fg_ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "fg_lpips": float(np.mean(lpips_vals)) if lpips_vals else None,
        "num_frames": len(per_frame),
        "per_frame": per_frame,
    }


def load_images_from_dir(img_dir: Path, dir_name: str) -> Dict[int, np.ndarray]:
    """Load all images from directory."""
    images = {}
    for img_path in sorted(img_dir.glob("*.png")):
        frame_num = get_frame_number(img_path.name)
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[frame_num] = img.astype(np.float32) / 255.0
    print(f"Loaded {len(images)} images from {dir_name}")
    return images


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute PSNR between prediction and ground truth."""
    mse = torch.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    psnr = -10.0 * torch.log(mse) / torch.log(torch.tensor(10.0))
    return psnr.item()


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute SSIM between prediction and ground truth."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_pred = torch.mean(pred)
    mu_gt = torch.mean(gt)
    sigma_pred = torch.var(pred)
    sigma_gt = torch.var(gt)
    sigma_pred_gt = torch.mean((pred - mu_pred) * (gt - mu_gt))
    ssim = ((2 * mu_pred * mu_gt + c1) * (2 * sigma_pred_gt + c2)) / \
           ((mu_pred ** 2 + mu_gt ** 2 + c1) * (sigma_pred + sigma_gt + c2))
    return ssim.item()


def evaluate_monofusion_directory(
    input_dir: Path,
    mask_dir: Optional[Path] = None,
    eval_full: bool = True,
    eval_fg: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate metrics for monofusion format directory.

    Monofusion format:
    - input_dir/cam_X/predicted/pred_frame_XXXX.png
    - input_dir/cam_X/ground_truth/gt_frame_XXXX.png

    Args:
        input_dir: Root directory containing cam_X folders
        mask_dir: Optional directory containing masks for foreground evaluation.
                  Masks are expected at mask_dir/{camera_name}/mask_{frame_idx:04d}.npy
        eval_full: Whether to evaluate full images
        eval_fg: Whether to evaluate foreground regions

    Returns:
        (metrics_dict, per_frame_results)
    """
    rendered, gt, camera_names = load_monofusion_images(input_dir)

    print(f"Total rendered images: {len(rendered)}, GT images: {len(gt)}")

    # Find common frames
    common_frames = set(rendered.keys()) & set(gt.keys())
    print(f"Common frames: {len(common_frames)}")

    if len(common_frames) == 0:
        print("Warning: No common frames found!")
        return {"avg_psnr": 0.0, "avg_ssim": 0.0}, []

    frames_per_cam = 200  # Assume 200 frames per camera

    # Initialize metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=False).to(device)

    # Calculate metrics for each frame
    per_frame_results = []
    lpips_values = []
    sorted_frames = sorted(common_frames)

    for frame_num in tqdm(sorted_frames, desc="Evaluating"):
        pred_img = rendered[frame_num]  # (H, W, 3)
        gt_img = gt[frame_num]          # (H, W, 3)

        # Resize if dimensions don't match
        if pred_img.shape[:2] != gt_img.shape[:2]:
            gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))

        # Convert to torch tensors (B, C, H, W)
        pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

        # Determine camera and local frame
        cam_idx = frame_num // frames_per_cam
        local_frame = frame_num % frames_per_cam

        frame_result = {
            "frame_idx": frame_num,
            "camera_idx": cam_idx,
            "local_frame": local_frame,
            "camera_name": camera_names[cam_idx] if cam_idx < len(camera_names) else f"cam_{cam_idx}",
        }

        # Update full image metrics
        if eval_full:
            psnr_metric.update(pred_tensor, gt_tensor)
            ssim_metric.update(pred_tensor, gt_tensor)

            # Calculate per-frame full image metrics
            psnr_val = compute_psnr(pred_tensor, gt_tensor)
            ssim_val = compute_ssim(pred_tensor, gt_tensor)
            lpips_val = lpips_metric(pred_tensor, gt_tensor).item()
            lpips_values.append(lpips_val)

            frame_result["psnr"] = psnr_val
            frame_result["ssim"] = ssim_val
            frame_result["lpips"] = lpips_val

        per_frame_results.append(frame_result)

    # Compute final metrics
    results = {}
    if eval_full:
        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        avg_lpips = float(np.mean(lpips_values)) if lpips_values else 0.0

        results["avg_psnr"] = avg_psnr
        results["avg_ssim"] = avg_ssim
        results["avg_lpips"] = avg_lpips

    # Foreground metrics (bbox crop + white fill)
    if eval_fg and mask_dir and mask_dir.exists():
        print("Computing foreground metrics (bbox crop + white fill)...")

        # Build pairs for each camera
        fg_metrics_by_cam = {}
        for cam_idx, camera_name in enumerate(camera_names):
            mask_cam_dir = mask_dir / camera_name
            if not mask_cam_dir.exists():
                continue

            # Build pairs for this camera
            pairs = []
            for frame_num in sorted_frames:
                frame_cam_idx = frame_num // frames_per_cam
                if frame_cam_idx != cam_idx:
                    continue

                frame_idx_in_cam = frame_num % frames_per_cam

                pred_img = rendered[frame_num]
                gt_img = gt[frame_num]
                if pred_img.shape[:2] != gt_img.shape[:2]:
                    gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))

                pairs.append((frame_idx_in_cam, pred_img, gt_img))

            if pairs:
                fg_result = compute_fg_metrics(pairs, mask_cam_dir, bbox_pad=0.4, device="cuda")
                if fg_result["num_frames"] > 0:
                    fg_metrics_by_cam[camera_name] = fg_result

        # Aggregate across cameras
        if fg_metrics_by_cam:
            all_psnr = []
            all_ssim = []
            all_lpips = []
            all_per_frame = []

            for cam_name, fg_result in fg_metrics_by_cam.items():
                if fg_result["fg_psnr"] is not None:
                    all_psnr.append(fg_result["fg_psnr"])
                    all_ssim.append(fg_result["fg_ssim"])
                    all_lpips.append(fg_result["fg_lpips"])
                    all_per_frame.extend(fg_result["per_frame"])

            if all_psnr:
                results["avg_fg_psnr"] = float(np.mean(all_psnr))
                results["avg_fg_ssim"] = float(np.mean(all_ssim))
                results["avg_fg_lpips"] = float(np.mean(all_lpips))
                results["fg_frames_evaluated"] = len(all_per_frame)
                print(f"Foreground metrics computed on {len(all_per_frame)} frames")

                # Add fg metrics to per_frame_results
                fg_lookup = {f["frame_idx"]: f for f in all_per_frame}
                for frame_result in per_frame_results:
                    frame_idx_in_cam = frame_result["local_frame"]
                    cam_idx = frame_result["camera_idx"]
                    # Create a unique key combining camera and frame
                    if frame_idx_in_cam in fg_lookup:
                        fg_data = fg_lookup[frame_idx_in_cam]
                        frame_result["fg_psnr"] = fg_data["fg_psnr"]
                        frame_result["fg_ssim"] = fg_data["fg_ssim"]
                        frame_result["fg_lpips"] = fg_data["fg_lpips"]

    return results, per_frame_results


def evaluate_directory(
    renders_dir: Path,
    gt_dir: Path,
    mask_dir: Optional[Path] = None,
    eval_full: bool = True,
    eval_fg: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate metrics for a single result directory.

    Args:
        renders_dir: Directory containing rendered images
        gt_dir: Directory containing ground truth images
        mask_dir: Optional directory containing masks for foreground evaluation.
                  Masks are expected at mask_dir/{camera_name}/mask_{frame_idx:04d}.npy
        eval_full: Whether to evaluate full images
        eval_fg: Whether to evaluate foreground regions

    Returns:
        (metrics_dict, per_frame_results)
    """
    if not renders_dir.exists():
        raise ValueError(f"Renders directory not found: {renders_dir}")
    if not gt_dir.exists():
        raise ValueError(f"GT directory not found: {gt_dir}")

    print(f"Loading rendered images from {renders_dir}...")
    rendered = load_images_from_dir(renders_dir, "renders")
    print(f"Loading GT images from {gt_dir}...")
    gt = load_images_from_dir(gt_dir, "gt")

    print(f"Rendered images: {len(rendered)}, GT images: {len(gt)}")

    # Find common frames
    common_frames = set(rendered.keys()) & set(gt.keys())
    print(f"Common frames: {len(common_frames)}")

    if len(common_frames) == 0:
        print("Warning: No common frames found!")
        return {"avg_psnr": 0.0, "avg_ssim": 0.0}, []

    # Discover camera names for mask loading
    camera_names = None
    frames_per_cam = 200  # Default frames per camera
    if eval_fg and mask_dir and mask_dir.exists():
        camera_names = discover_camera_names(mask_dir)
        print(f"Found {len(camera_names)} cameras in mask directory: {camera_names[:3]}...")

    # Initialize metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=False).to(device)

    # Calculate metrics for each frame
    per_frame_results = []
    lpips_values = []
    sorted_frames = sorted(common_frames)

    for frame_num in tqdm(sorted_frames, desc="Evaluating"):
        pred_img = rendered[frame_num]  # (H, W, 3)
        gt_img = gt[frame_num]          # (H, W, 3)

        # Resize if dimensions don't match
        if pred_img.shape[:2] != gt_img.shape[:2]:
            gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))

        # Convert to torch tensors (B, C, H, W)
        pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)  # (1, 3, H, W)
        gt_tensor = torch.from_numpy(gt_img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)      # (1, 3, H, W)

        frame_result = {
            "frame_idx": frame_num,
            "actual_frame": render_idx_to_frame(frame_num) if mask_dir else None,
        }

        # Update full image metrics
        if eval_full:
            psnr_metric.update(pred_tensor, gt_tensor)
            ssim_metric.update(pred_tensor, gt_tensor)

            # Calculate per-frame full image metrics
            psnr_val = compute_psnr(pred_tensor, gt_tensor)
            ssim_val = compute_ssim(pred_tensor, gt_tensor)
            lpips_val = lpips_metric(pred_tensor, gt_tensor).item()
            lpips_values.append(lpips_val)

            frame_result["psnr"] = psnr_val
            frame_result["ssim"] = ssim_val
            frame_result["lpips"] = lpips_val

        per_frame_results.append(frame_result)

    # Compute final metrics
    results = {}
    if eval_full:
        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        avg_lpips = float(np.mean(lpips_values)) if lpips_values else 0.0

        results["avg_psnr"] = avg_psnr
        results["avg_ssim"] = avg_ssim
        results["avg_lpips"] = avg_lpips

    # Foreground metrics (bbox crop + white fill)
    if eval_fg and mask_dir and mask_dir.exists() and camera_names:
        print("Computing foreground metrics (bbox crop + white fill)...")

        # Build pairs for each camera
        fg_metrics_by_cam = {}
        for cam_idx, camera_name in enumerate(camera_names):
            # Get mask directory for this camera
            mask_cam_dir = mask_dir / camera_name
            if not mask_cam_dir.exists():
                continue

            # Build pairs for this camera
            pairs = []
            for frame_num in sorted_frames:
                # Determine if this frame belongs to this camera
                frame_cam_idx = frame_num // frames_per_cam
                if frame_cam_idx != cam_idx:
                    continue

                # Get frame index within camera
                frame_idx_in_cam = frame_num % frames_per_cam

                pred_img = rendered[frame_num]
                gt_img = gt[frame_num]
                if pred_img.shape[:2] != gt_img.shape[:2]:
                    gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))

                pairs.append((frame_idx_in_cam, pred_img, gt_img))

            if pairs:
                fg_result = compute_fg_metrics(pairs, mask_cam_dir, bbox_pad=0.4, device="cuda")
                if fg_result["num_frames"] > 0:
                    fg_metrics_by_cam[camera_name] = fg_result

        # Aggregate across cameras
        if fg_metrics_by_cam:
            all_psnr = []
            all_ssim = []
            all_lpips = []
            all_per_frame = []

            for cam_name, fg_result in fg_metrics_by_cam.items():
                if fg_result["fg_psnr"] is not None:
                    all_psnr.append(fg_result["fg_psnr"])
                    all_ssim.append(fg_result["fg_ssim"])
                    all_lpips.append(fg_result["fg_lpips"])
                    all_per_frame.extend(fg_result["per_frame"])

            if all_psnr:
                results["avg_fg_psnr"] = float(np.mean(all_psnr))
                results["avg_fg_ssim"] = float(np.mean(all_ssim))
                results["avg_fg_lpips"] = float(np.mean(all_lpips))
                results["fg_frames_evaluated"] = len(all_per_frame)
                print(f"Foreground metrics computed on {len(all_per_frame)} frames")

                # Add fg metrics to per_frame_results
                fg_lookup = {f["frame_idx"]: f for f in all_per_frame}
                for frame_result in per_frame_results:
                    frame_idx_in_cam = frame_result["frame_idx"] % frames_per_cam
                    if frame_idx_in_cam in fg_lookup:
                        fg_data = fg_lookup[frame_idx_in_cam]
                        frame_result["fg_psnr"] = fg_data["fg_psnr"]
                        frame_result["fg_ssim"] = fg_data["fg_ssim"]
                        frame_result["fg_lpips"] = fg_data["fg_lpips"]

    return results, per_frame_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UniFusion rendering results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to result directory. Supports two formats: "
             "1) UniFusion format: /path/to/test/ours_XXXXX with renders/ and gt/ subdirs; "
             "2) Monofusion format: /path/to/monofusion_results/scene_name with cam_X/predicted/ and cam_X/ground_truth/ subdirs. "
             "Not needed if --renders_dir and --gt_dir are provided."
    )
    parser.add_argument(
        "--renders_dir",
        type=str,
        default=None,
        help="Path to rendered images directory. Can be used independently or with --input_dir."
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Path to ground truth images directory. Can be used independently or with --input_dir."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format). If None, print to stdout"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Directory containing SAM masks for foreground evaluation. "
             "If not provided, only full-image metrics are computed."
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default=None,
        help="Sequence name used with --mask_root"
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default=None,
        help="Optional root containing _<seq_name>/sam_v2_dyn_mask"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="both",
        choices=["full", "foreground", "both"],
        help="Evaluation mode: 'full' for full images only, 'foreground' for foreground only, 'both' for both"
    )
    args = parser.parse_args()

    # Validate arguments
    has_input_dir = args.input_dir is not None
    has_renders_dir = args.renders_dir is not None
    has_gt_dir = args.gt_dir is not None

    if not has_input_dir and not (has_renders_dir and has_gt_dir):
        parser.error("Must provide either --input_dir OR both --renders_dir and --gt_dir")

    if has_renders_dir != has_gt_dir:
        parser.error("--renders_dir and --gt_dir must be provided together")

    # Determine mask directory
    mask_dir = None
    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
    elif args.seq_name and args.mask_root:
        mask_dir = Path(args.mask_root) / f"_{args.seq_name}" / "sam_v2_dyn_mask"
    elif args.seq_name:
        parser.error("--seq_name requires --mask_root (or pass --mask_dir directly)")

    # Determine evaluation mode
    eval_full = args.eval_mode in ["full", "both"]
    eval_fg = args.eval_mode in ["foreground", "both"]

    # Check if using independent renders/gt dirs
    if has_renders_dir and has_gt_dir:
        renders_dir = Path(args.renders_dir)
        gt_dir = Path(args.gt_dir)

        if not renders_dir.exists():
            raise ValueError(f"Renders directory not found: {renders_dir}")
        if not gt_dir.exists():
            raise ValueError(f"GT directory not found: {gt_dir}")

        print(f"Evaluating with independent directories:")
        print(f"  Renders: {renders_dir}")
        print(f"  GT: {gt_dir}")
        print(f"  Eval mode: {args.eval_mode}")
        if eval_fg and mask_dir:
            print(f"  Masks: {mask_dir}")
            if not mask_dir.exists():
                print(f"  Warning: Mask directory not found: {mask_dir}")
        elif eval_fg and not mask_dir:
            print(f"  Warning: Foreground evaluation requested but no mask_dir provided")

        results, per_frame = evaluate_directory(
            renders_dir, gt_dir, mask_dir,
            eval_full=eval_full, eval_fg=eval_fg
        )
    else:
        # Use input_dir mode
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        # Check if input is in monofusion format
        is_mono = is_monofusion_format(input_dir)

        if is_mono:
            print(f"Detected monofusion format: {input_dir}")
            print(f"  Eval mode: {args.eval_mode}")
            print(f"  Masks: {mask_dir}")
            if mask_dir and not mask_dir.exists():
                print(f"  Warning: Mask directory not found: {mask_dir}")

            results, per_frame = evaluate_monofusion_directory(
                input_dir, mask_dir,
                eval_full=eval_full, eval_fg=eval_fg
            )
        else:
            renders_dir = input_dir / "renders"
            gt_dir = input_dir / "gt"

            print(f"Evaluating: {input_dir}")
            print(f"  Renders: {renders_dir}")
            print(f"  GT: {gt_dir}")
            print(f"  Eval mode: {args.eval_mode}")
            if eval_fg and mask_dir:
                print(f"  Masks: {mask_dir}")
                if not mask_dir.exists():
                    print(f"  Warning: Mask directory not found: {mask_dir}")
            elif eval_fg and not mask_dir:
                print(f"  Warning: Foreground evaluation requested but no mask_dir provided")

            results, per_frame = evaluate_directory(
                renders_dir, gt_dir, mask_dir,
                eval_full=eval_full, eval_fg=eval_fg
            )

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    if eval_full:
        print(f"\n[Full Image Metrics]")
        print(f"  PSNR  = {results['avg_psnr']:.2f} dB")
        print(f"  SSIM  = {results['avg_ssim']:.4f}")
        print(f"  LPIPS = {results['avg_lpips']:.4f}")

    if "avg_fg_psnr" in results:
        print(f"\n[Foreground Metrics]")
        print(f"  PSNR  = {results['avg_fg_psnr']:.2f} dB")
        print(f"  SSIM  = {results['avg_fg_ssim']:.4f}")
        print(f"  LPIPS = {results['avg_fg_lpips']:.4f}")
        print(f"  Frames evaluated: {results['fg_frames_evaluated']}")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)

        # Build config based on which mode was used
        config = {
            "eval_mode": args.eval_mode,
            "mask_dir": str(mask_dir) if mask_dir else None,
        }

        if has_renders_dir and has_gt_dir:
            config["renders_dir"] = str(args.renders_dir)
            config["gt_dir"] = str(args.gt_dir)
        else:
            config["input_dir"] = str(args.input_dir)

        output_data = {
            "summary": results,
            "per_frame": per_frame,
            "config": config
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
