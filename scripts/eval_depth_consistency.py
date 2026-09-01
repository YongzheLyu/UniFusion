#!/usr/bin/env python3
"""Evaluate temporal and cross-view consistency of depth maps.

The script supports one depth output root per run. It is intended for both
MAtCha priors-style depth folders and rendered-depth folders.
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
EPS = 1e-6


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


@dataclass
class DepthRecord:
    camera_id: int
    frame_id: int
    path: Path


@dataclass
class DepthDataset:
    depth_dir: Path
    layout: str
    camera_ids: List[int]
    frame_ids: List[int]
    records: Dict[Tuple[int, int], Path]


@dataclass
class RunningStats:
    count: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_rel: float = 0.0
    sum_penalty: float = 0.0
    violations: int = 0
    threshold_violations: Optional[Dict[str, int]] = None

    def add_errors(self, abs_err: np.ndarray, denom: Optional[np.ndarray] = None):
        abs_err = np.asarray(abs_err, dtype=np.float64)
        if abs_err.size == 0:
            return
        valid = np.isfinite(abs_err)
        if denom is not None:
            denom = np.asarray(denom, dtype=np.float64)
            valid &= np.isfinite(denom) & (np.abs(denom) > EPS)
        if not np.any(valid):
            return
        vals = abs_err[valid]
        self.count += int(vals.size)
        self.sum_abs += float(vals.sum())
        self.sum_sq += float((vals * vals).sum())
        if denom is not None:
            self.sum_rel += float((vals / np.maximum(np.abs(denom[valid]), EPS)).sum())

    def add_error_sums(self, count: int, sum_abs: float, sum_sq: float, sum_rel: float = 0.0):
        if count <= 0:
            return
        self.count += int(count)
        self.sum_abs += float(sum_abs)
        self.sum_sq += float(sum_sq)
        self.sum_rel += float(sum_rel)

    def add_order(self, penalty: np.ndarray, violations: np.ndarray, threshold_violations: Optional[Dict[str, np.ndarray]] = None):
        penalty = np.asarray(penalty, dtype=np.float64)
        violations = np.asarray(violations, dtype=bool)
        valid = np.isfinite(penalty)
        if not np.any(valid):
            return
        self.count += int(valid.sum())
        self.sum_penalty += float(np.maximum(penalty[valid], 0.0).sum())
        self.violations += int(violations[valid].sum())
        if threshold_violations is not None:
            if self.threshold_violations is None:
                self.threshold_violations = {}
            for key, values in threshold_violations.items():
                values = np.asarray(values, dtype=bool)
                self.threshold_violations[key] = self.threshold_violations.get(key, 0) + int(values[valid].sum())

    def add_order_sums(
        self,
        count: int,
        sum_penalty: float,
        violations: int,
        threshold_violations: Optional[Dict[str, int]] = None,
    ):
        if count <= 0:
            return
        self.count += int(count)
        self.sum_penalty += float(sum_penalty)
        self.violations += int(violations)
        if threshold_violations is not None:
            if self.threshold_violations is None:
                self.threshold_violations = {}
            for key, value in threshold_violations.items():
                self.threshold_violations[key] = self.threshold_violations.get(key, 0) + int(value)

    def as_error_dict(self):
        if self.count == 0:
            return {"count": 0, "mean": None, "rmse": None, "absrel": None}
        return {
            "count": self.count,
            "mean": self.sum_abs / self.count,
            "rmse": math.sqrt(self.sum_sq / self.count),
            "absrel": self.sum_rel / self.count,
        }

    def as_order_dict(self):
        if self.count == 0:
            return {"count": 0, "violation_rate": None, "penalty_mean": None}
        return {
            "count": self.count,
            "violations": self.violations,
            "violation_rate": self.violations / self.count,
            "penalty_mean": self.sum_penalty / self.count,
            **{
                f"violation_rate@{key}": value / self.count
                for key, value in sorted((self.threshold_violations or {}).items(), key=lambda item: float(item[0]))
            },
        }


@dataclass
class RunningScalarStats:
    count: int = 0
    sum_value: float = 0.0

    def add(self, value: Optional[float]):
        if value is None or not np.isfinite(value):
            return
        self.count += 1
        self.sum_value += float(value)

    def as_dict(self):
        if self.count == 0:
            return {"count": 0, "mean": None}
        return {"count": self.count, "mean": self.sum_value / self.count}


def natural_key(value: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", value)]


def numeric_tokens(path: Path) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", path.stem)]


def parse_cam_frame_from_name(path: Path) -> Optional[Tuple[int, int]]:
    nums = numeric_tokens(path)
    if len(nums) >= 2 and path.stem.startswith("cam"):
        return nums[0], nums[1]
    if len(nums) >= 2 and "cam" in path.stem.lower():
        return nums[-2], nums[-1]
    return None


def parse_frame_from_name(path: Path) -> Optional[int]:
    nums = numeric_tokens(path)
    if not nums:
        return None
    return nums[0]


def parse_camera_id_from_dir(path: Path) -> Optional[int]:
    nums = [int(x) for x in re.findall(r"\d+", path.name)]
    if not nums:
        return None
    return nums[-1]


def discover_depth_records(depth_dir: Path, layout: str) -> DepthDataset:
    depth_dir = depth_dir.resolve()
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory does not exist: {depth_dir}")

    if layout == "auto":
        if (depth_dir / "depths").is_dir() and list((depth_dir / "depths").glob("*.npy")):
            layout = "priors"
        elif list(depth_dir.glob("cam_*.npy")):
            layout = "priors"
        elif any((d / "depth").is_dir() or (d / "depths").is_dir() for d in depth_dir.iterdir() if d.is_dir()):
            layout = "render_by_cam"
        elif (depth_dir / "depth").is_dir() and list((depth_dir / "depth").glob("*.npy")):
            layout = "render_flat"
        elif list(depth_dir.glob("*.npy")):
            layout = "render_flat"
        else:
            raise ValueError(f"Could not infer depth layout under {depth_dir}")

    records: List[DepthRecord] = []
    if layout == "priors":
        root = depth_dir / "depths" if (depth_dir / "depths").is_dir() else depth_dir
        for path in sorted(root.glob("*.npy"), key=lambda p: natural_key(p.name)):
            parsed = parse_cam_frame_from_name(path)
            if parsed is None:
                continue
            records.append(DepthRecord(parsed[0], parsed[1], path))
    elif layout == "render_by_cam":
        for cam_dir in sorted([d for d in depth_dir.iterdir() if d.is_dir()], key=lambda p: natural_key(p.name)):
            camera_id = parse_camera_id_from_dir(cam_dir)
            if camera_id is None:
                continue
            depth_root = cam_dir / "depth"
            if not depth_root.is_dir():
                depth_root = cam_dir / "depths"
            if not depth_root.is_dir():
                continue
            for path in sorted(depth_root.glob("*.npy"), key=lambda p: natural_key(p.name)):
                frame_id = parse_frame_from_name(path)
                if frame_id is not None:
                    records.append(DepthRecord(camera_id, frame_id, path))
    elif layout == "render_flat":
        root = depth_dir / "depth" if (depth_dir / "depth").is_dir() else depth_dir
        for path in sorted(root.glob("*.npy"), key=lambda p: natural_key(p.name)):
            frame_id = parse_frame_from_name(path)
            if frame_id is not None:
                records.append(DepthRecord(0, frame_id, path))
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if not records:
        raise ValueError(f"No depth .npy records found for layout={layout} in {depth_dir}")

    records_map = {(r.camera_id, r.frame_id): r.path for r in records}
    camera_ids = sorted({r.camera_id for r in records})
    frame_ids = sorted({r.frame_id for r in records})
    return DepthDataset(depth_dir, layout, camera_ids, frame_ids, records_map)


def load_depth(
    path: Path,
    target_hw: Optional[Tuple[int, int]] = None,
    mode: str = "nearest",
    scale_divisor: float = 1.0,
    inverse_depth: bool = False,
) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map at {path}, got shape {depth.shape}")
    if inverse_depth:
        depth = 1.0 / np.maximum(depth, EPS)
    if scale_divisor != 1.0:
        depth = depth / float(scale_divisor)
    if target_hw is not None and depth.shape != target_hw:
        tensor = torch.from_numpy(depth)[None, None]
        if mode == "bilinear":
            resized = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
        else:
            resized = F.interpolate(tensor, size=target_hw, mode="nearest")
        depth = resized[0, 0].numpy().astype(np.float32)
    return depth


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def interpolate_depth_tensor(depth: torch.Tensor, target_hw: Tuple[int, int], mode: str) -> torch.Tensor:
    if tuple(depth.shape[-2:]) == tuple(target_hw):
        return depth
    tensor = depth[None, None].float()
    if mode == "bilinear":
        resized = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    else:
        resized = F.interpolate(tensor, size=target_hw, mode="nearest")
    return resized[0, 0]


def load_depth_tensor(
    path: Path,
    device: torch.device,
    target_hw: Optional[Tuple[int, int]] = None,
    mode: str = "nearest",
    scale_divisor: float = 1.0,
    inverse_depth: bool = False,
) -> torch.Tensor:
    depth = torch.from_numpy(np.squeeze(np.load(path).astype(np.float32)))
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map at {path}, got shape {tuple(depth.shape)}")
    if inverse_depth:
        depth = 1.0 / torch.clamp(depth, min=EPS)
    if scale_divisor != 1.0:
        depth = depth / float(scale_divisor)
    depth = depth.to(device=device, non_blocking=True)
    if target_hw is not None:
        depth = interpolate_depth_tensor(depth, target_hw, mode)
    return depth


class DepthTensorCache:
    def __init__(self, dataset: DepthDataset, args, device: torch.device):
        self.dataset = dataset
        self.args = args
        self.device = device
        self.cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self.cache_device = args.depth_cache_device
        if self.cache_device == "auto":
            self.cache_device = "cuda" if device.type == "cuda" and self._fits_gpu_cache_budget() else "cpu"
        if self.cache_device == "cuda" and device.type != "cuda":
            self.cache_device = "cpu"

    def _fits_gpu_cache_budget(self) -> bool:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return False
        try:
            sample_path = next(iter(self.dataset.records.values()))
            sample = np.squeeze(np.load(sample_path, mmap_mode="r"))
            bytes_per_depth = int(np.prod(sample.shape) * np.dtype(np.float32).itemsize)
            estimated = bytes_per_depth * len(self.dataset.records)
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            budget = min(8 * 1024**3, int(free_bytes * 0.35))
            return estimated <= budget
        except Exception:
            return False

    def get(self, cam: int, frame: int, target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        key = (cam, frame)
        use_cache = self.cache_device in {"cpu", "cuda"}
        if key not in self.cache:
            cache_device = self.device if self.cache_device == "cuda" else torch.device("cpu")
            self.cache[key] = load_depth_tensor(
                self.dataset.records[key],
                cache_device,
                mode=self.args.depth_resize_mode,
                scale_divisor=self.args.depth_scale_divisor,
                inverse_depth=self.args.inverse_depth,
            )
        depth = self.cache[key].to(self.device, non_blocking=True) if use_cache else load_depth_tensor(
            self.dataset.records[key],
            self.device,
            mode=self.args.depth_resize_mode,
            scale_divisor=self.args.depth_scale_divisor,
            inverse_depth=self.args.inverse_depth,
        )
        if target_hw is not None:
            depth = interpolate_depth_tensor(depth, target_hw, self.args.depth_resize_mode)
        return depth

    def clear(self):
        if self.cache_device != "cuda":
            self.cache.clear()


def load_pose_metadata(pose_file: Path):
    with open(pose_file, "r") as f:
        data = json.load(f)
    poses = np.asarray(data["poses"], dtype=np.float64)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape [V,4,4], got {poses.shape}")
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"intrinsics must have shape [V,3,3], got {intrinsics.shape}")
    camera_names = data.get("camera_names", [f"cam{i:02d}" for i in range(len(poses))])
    return poses, np.linalg.inv(poses), intrinsics, camera_names


def scaled_intrinsic(K: np.ndarray, depth_hw: Tuple[int, int], image_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if image_hw is None or image_hw == depth_hw:
        return K.astype(np.float64)
    dh, dw = depth_hw
    ih, iw = image_hw
    out = K.astype(np.float64).copy()
    out[0, :] *= dw / iw
    out[1, :] *= dh / ih
    return out


def list_image_dirs(image_dir: Path) -> List[Path]:
    if image_dir is None:
        return []
    return sorted([d for d in image_dir.iterdir() if d.is_dir()], key=lambda p: natural_key(p.name))


def build_image_index(image_dir: Optional[Path], camera_ids: Sequence[int], camera_names: Sequence[str]):
    if image_dir is None:
        return {}, {}
    image_dir = image_dir.resolve()
    dirs = list_image_dirs(image_dir)
    frame_dirs = [d for d in dirs if d.name.startswith("frame_")]
    if frame_dirs and any((d / "mast3r_sfm" / "images").is_dir() for d in frame_dirs):
        index: Dict[int, Dict[int, Path]] = {cam_id: {} for cam_id in camera_ids}
        image_roots = []
        for frame_dir in sorted(frame_dirs, key=lambda p: natural_key(p.name)):
            image_root = frame_dir / "mast3r_sfm" / "images"
            if not image_root.is_dir():
                continue
            image_roots.append(image_root)
            for path in sorted(image_root.iterdir(), key=lambda p: natural_key(p.name)):
                if path.suffix.lower() not in IMAGE_EXTS:
                    continue
                nums = numeric_tokens(path)
                if len(nums) < 2:
                    continue
                cam_id = nums[0]
                frame_id = nums[-1]
                if cam_id in index:
                    index[cam_id].setdefault(frame_id, path)
        return index, {cam_id: str(image_dir) for cam_id in camera_ids}

    if not dirs:
        files = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        return {camera_ids[0]: index_image_files(files)} if len(camera_ids) == 1 else {}, {camera_ids[0]: str(image_dir)}

    mapping: Dict[int, Path] = {}
    used = set()
    for idx, cam_id in enumerate(camera_ids):
        candidates = [f"cam{cam_id:02d}", f"cam{cam_id:04d}", f"camera_{cam_id}", f"camera_{cam_id + 1}"]
        if idx < len(camera_names):
            candidates.append(camera_names[idx])
        for name in candidates:
            cand = image_dir / name
            if cand.is_dir() and cand not in used:
                mapping[cam_id] = cand
                used.add(cand)
                break

    remaining_cams = [c for c in camera_ids if c not in mapping]
    remaining_dirs = [d for d in dirs if d not in used]
    for cam_id, cam_dir in zip(remaining_cams, remaining_dirs):
        mapping[cam_id] = cam_dir

    index = {}
    for cam_id, cam_dir in mapping.items():
        files = [p for p in cam_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        index[cam_id] = index_image_files(files)
    return index, {k: str(v) for k, v in mapping.items()}


def index_image_files(files: Iterable[Path]) -> Dict[int, Path]:
    out = {}
    for path in sorted(files, key=lambda p: natural_key(p.name)):
        nums = numeric_tokens(path)
        if not nums:
            continue
        frame_id = nums[-1]
        out.setdefault(frame_id, path)
        out.setdefault(nums[0], path)
    return out


def resolve_image_pair(image_index: Dict[int, Path], frames_sorted: List[int], frame_a: int, frame_b: int):
    if frame_a in image_index and frame_b in image_index:
        return image_index[frame_a], image_index[frame_b]
    try:
        ia = frames_sorted.index(frame_a)
        ib = frames_sorted.index(frame_b)
    except ValueError:
        return None, None
    paths_sorted = sorted(image_index.values(), key=lambda p: natural_key(p.name))
    if ia < len(paths_sorted) and ib < len(paths_sorted):
        return paths_sorted[ia], paths_sorted[ib]
    return None, None


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_flow(flow: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = flow.shape[:2]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return flow.astype(np.float32)
    tensor = torch.from_numpy(flow).permute(2, 0, 1)[None].float()
    resized = F.interpolate(tensor, size=(th, tw), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    resized = resized.numpy()
    resized[..., 0] *= tw / w
    resized[..., 1] *= th / h
    return resized.astype(np.float32)


def get_raft_flow(img_a: Path, img_b: Path, cache_path: Optional[Path], model_type: str, device: str):
    if cache_path is not None and cache_path.exists():
        try:
            return np.load(cache_path)["flow"].astype(np.float32)
        except (EOFError, OSError, ValueError, KeyError):
            cache_path.unlink(missing_ok=True)
    from compute_optical_flow import FlowEstimator

    estimator = get_raft_flow.estimators.get((model_type, device))
    if estimator is None:
        torch_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        estimator = FlowEstimator(model_type=model_type, device=torch_device)
        get_raft_flow.estimators[(model_type, device)] = estimator
    flow = estimator.compute_pairs([(load_image(img_a), load_image(img_b))], batch_size=1)[0].astype(np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, flow=flow)
    return flow


get_raft_flow.estimators = {}


def get_raft_flows_batch(items: Sequence[Tuple[Path, Path, Optional[Path]]], model_type: str, device: str, batch_size: int):
    flows: List[Optional[np.ndarray]] = [None] * len(items)
    missing = []
    for idx, (img_a, img_b, cache_path) in enumerate(items):
        if cache_path is not None and cache_path.exists():
            try:
                flows[idx] = np.load(cache_path)["flow"].astype(np.float32)
                continue
            except (EOFError, OSError, ValueError, KeyError):
                cache_path.unlink(missing_ok=True)
        missing.append((idx, img_a, img_b, cache_path))

    if missing:
        from compute_optical_flow import FlowEstimator

        estimator = get_raft_flow.estimators.get((model_type, device))
        if estimator is None:
            torch_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
            estimator = FlowEstimator(model_type=model_type, device=torch_device)
            get_raft_flow.estimators[(model_type, device)] = estimator

        for start in range(0, len(missing), max(1, batch_size)):
            batch = missing[start:start + max(1, batch_size)]
            pil_pairs = [(load_image(img_a), load_image(img_b)) for _, img_a, img_b, _ in batch]
            batch_flows = estimator.compute_pairs(pil_pairs, batch_size=max(1, batch_size))
            for (idx, _, _, cache_path), flow in zip(batch, batch_flows):
                flow = flow.astype(np.float32)
                flows[idx] = flow
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache_path, flow=flow)

    return [flow for flow in flows if flow is not None]


def make_grid(height: int, width: int, stride: int) -> np.ndarray:
    ys = np.arange(0, height, stride, dtype=np.float32)
    xs = np.arange(0, width, stride, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


_TORCH_GRID_CACHE: Dict[Tuple[int, int, int, str], torch.Tensor] = {}


def make_grid_torch(height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
    key = (height, width, max(stride, 1), str(device))
    if key not in _TORCH_GRID_CACHE:
        ys = torch.arange(0, height, max(stride, 1), device=device, dtype=torch.float32)
        xs = torch.arange(0, width, max(stride, 1), device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        _TORCH_GRID_CACHE[key] = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return _TORCH_GRID_CACHE[key]


def bilinear_sample_torch(image: torch.Tensor, points_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    squeeze_batch = False
    if image.ndim == 2:
        image_bchw = image[None, None]
        squeeze_batch = True
    elif image.ndim == 3:
        image_bchw = image[:, None]
    elif image.ndim == 4 and image.shape[-1] <= 4:
        image_bchw = image.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported image tensor shape for sampling: {tuple(image.shape)}")

    b, c, h, w = image_bchw.shape
    if points_xy.ndim == 2:
        points = points_xy[None].expand(b, -1, -1)
    elif points_xy.ndim == 3:
        points = points_xy
    else:
        raise ValueError(f"Unsupported points tensor shape: {tuple(points_xy.shape)}")
    if points.shape[0] != b:
        if points.shape[0] == 1:
            points = points.expand(b, -1, -1)
        else:
            raise ValueError(f"Point batch {points.shape[0]} does not match image batch {b}")

    x = points[..., 0]
    y = points[..., 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    if w > 1:
        x_norm = 2.0 * x / (w - 1) - 1.0
    else:
        x_norm = torch.zeros_like(x)
    if h > 1:
        y_norm = 2.0 * y / (h - 1) - 1.0
    else:
        y_norm = torch.zeros_like(y)
    grid = torch.stack([x_norm, y_norm], dim=-1).view(b, -1, 1, 2)
    sampled = F.grid_sample(image_bchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled[..., 0].permute(0, 2, 1)
    if c == 1:
        sampled = sampled[..., 0]
    if squeeze_batch:
        return sampled[0], valid[0]
    return sampled, valid


def resize_flow_tensor(flow: np.ndarray, target_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    h, w = flow.shape[:2]
    th, tw = target_hw
    tensor = torch.from_numpy(flow).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    if (h, w) != (th, tw):
        tensor = F.interpolate(tensor, size=(th, tw), mode="bilinear", align_corners=False)
        tensor[:, 0] *= tw / w
        tensor[:, 1] *= th / h
    return tensor[0].permute(1, 2, 0)


def add_torch_error_stats(stats: RunningStats, err: torch.Tensor, denom: Optional[torch.Tensor] = None):
    if err.numel() == 0:
        return
    err64 = err.double()
    if denom is None:
        stats.add_error_sums(err64.numel(), err64.sum().item(), (err64 * err64).sum().item(), 0.0)
    else:
        denom64 = denom.double()
        rel = err64 / torch.clamp(denom64.abs(), min=EPS)
        stats.add_error_sums(err64.numel(), err64.sum().item(), (err64 * err64).sum().item(), rel.sum().item())


def add_torch_order_stats(
    stats: RunningStats,
    penalty: torch.Tensor,
    violations: torch.Tensor,
    threshold_violations: Optional[Dict[str, torch.Tensor]] = None,
):
    if penalty.numel() == 0:
        return
    threshold_counts = None
    if threshold_violations is not None:
        threshold_counts = {key: int(values.sum().item()) for key, values in threshold_violations.items()}
    stats.add_order_sums(
        int(penalty.numel()),
        torch.clamp(penalty.double(), min=0.0).sum().item(),
        int(violations.sum().item()),
        threshold_counts,
    )


def bilinear_sample(image: np.ndarray, points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x0 = np.floor(np.clip(x, 0, w - 1)).astype(np.int64)
    y0 = np.floor(np.clip(y, 0, h - 1)).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wa = ((x1 - x) * (y1 - y))
    wb = ((x - x0) * (y1 - y))
    wc = ((x1 - x) * (y - y0))
    wd = ((x - x0) * (y - y0))
    same_x = x0 == x1
    same_y = y0 == y1
    wa[same_x & same_y] = 1.0
    wb[same_x] = 0.0
    wc[same_y] = 0.0
    wd[same_x | same_y] = 0.0

    if image.ndim == 2:
        sampled = (
            wa * image[y0, x0]
            + wb * image[y0, x1]
            + wc * image[y1, x0]
            + wd * image[y1, x1]
        )
    else:
        sampled = (
            wa[..., None] * image[y0, x0]
            + wb[..., None] * image[y0, x1]
            + wc[..., None] * image[y1, x0]
            + wd[..., None] * image[y1, x1]
        )
    return sampled, valid


def backproject(points_xy: np.ndarray, depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    n = points_xy.shape[0]
    pixels = np.concatenate([points_xy, np.ones((n, 1), dtype=np.float64)], axis=1)
    xyz_cam = (pixels @ np.linalg.inv(K).T) * depth[:, None]
    xyz_h = np.concatenate([xyz_cam, np.ones((n, 1), dtype=np.float64)], axis=1)
    return (xyz_h @ c2w.T)[:, :3]


def project(points_world: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = points_world.shape[0]
    xyz_h = np.concatenate([points_world, np.ones((n, 1), dtype=np.float64)], axis=1)
    xyz_cam = (xyz_h @ w2c.T)[:, :3]
    z = xyz_cam[:, 2]
    pix_h = xyz_cam @ K.T
    xy = pix_h[:, :2] / np.maximum(pix_h[:, 2:3], EPS)
    return xy, z


def backproject_torch(points_xy: torch.Tensor, depth: torch.Tensor, K_inv: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((*points_xy.shape[:-1], 1), device=points_xy.device, dtype=points_xy.dtype)
    pixels = torch.cat([points_xy, ones], dim=-1)
    xyz_cam = (pixels @ K_inv.transpose(-1, -2)) * depth[..., None]
    xyz_h = torch.cat([xyz_cam, ones], dim=-1)
    return (xyz_h @ c2w.transpose(-1, -2))[..., :3]


def project_torch(points_world: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((*points_world.shape[:-1], 1), device=points_world.device, dtype=points_world.dtype)
    xyz_h = torch.cat([points_world, ones], dim=-1)
    xyz_cam = (xyz_h @ w2c.transpose(-1, -2))[..., :3]
    z = xyz_cam[..., 2]
    pix_h = xyz_cam @ K.transpose(-1, -2)
    xy = pix_h[..., :2] / torch.clamp(pix_h[..., 2:3], min=EPS)
    return xy, z


def scaled_intrinsic_torch(
    K: np.ndarray,
    depth_hw: Tuple[int, int],
    image_hw: Optional[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(scaled_intrinsic(K, depth_hw, image_hw)).to(device=device, dtype=torch.float32)


def round_splat_depth(
    depth_src: np.ndarray,
    depth_tgt: np.ndarray,
    K_src: np.ndarray,
    K_tgt: np.ndarray,
    c2w_src: np.ndarray,
    w2c_tgt: np.ndarray,
    stride: int,
    args,
) -> np.ndarray:
    grid = make_grid(depth_src.shape[0], depth_src.shape[1], stride)
    z_src, src_valid = bilinear_sample(depth_src, grid)
    valid = src_valid & np.isfinite(z_src) & (z_src > args.min_depth)
    if args.max_depth is not None:
        valid &= z_src < args.max_depth

    warped = np.zeros(depth_tgt.shape, dtype=np.float32)
    if not np.any(valid):
        return warped

    points_world = backproject(grid[valid], z_src[valid], K_src, c2w_src)
    uv_tgt, z_proj = project(points_world, K_tgt, w2c_tgt)
    coords = np.round(uv_tgt).astype(np.int64)
    h, w = depth_tgt.shape
    target_valid = (
        np.isfinite(z_proj)
        & (z_proj > args.min_depth)
        & (coords[:, 0] >= 0)
        & (coords[:, 0] < w)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < h)
    )
    if args.max_depth is not None:
        target_valid &= z_proj < args.max_depth
    coords = coords[target_valid]
    z_proj = z_proj[target_valid]
    warped[coords[:, 1], coords[:, 0]] = z_proj.astype(np.float32)
    return warped


def select_cameras_and_frames(dataset: DepthDataset, args):
    camera_ids = dataset.camera_ids
    if args.camera_ids:
        requested = set(args.camera_ids)
        camera_ids = [c for c in camera_ids if c in requested]
    frame_ids = dataset.frame_ids
    if args.frame_start is not None:
        frame_ids = [f for f in frame_ids if f >= args.frame_start]
    if args.frame_end is not None:
        frame_ids = [f for f in frame_ids if f <= args.frame_end]
    frame_ids = frame_ids[:: max(args.frame_stride, 1)]
    complete_frames = []
    for f in frame_ids:
        if all((c, f) in dataset.records for c in camera_ids):
            complete_frames.append(f)
    return camera_ids, complete_frames


def camera_pose_index(camera_ids: Sequence[int], n_poses: int) -> Dict[int, int]:
    if len(camera_ids) > n_poses:
        raise ValueError(f"Depth has {len(camera_ids)} cameras but pose file has {n_poses} poses")
    return {cam_id: idx for idx, cam_id in enumerate(sorted(camera_ids))}


def evaluate_flow_depth(dataset, camera_ids, frame_ids, image_index, args):
    start_time = time.time()
    device = resolve_torch_device(args.device)
    depth_cache = DepthTensorCache(dataset, args, device)
    overall = RunningStats()
    per_camera = defaultdict(RunningStats)
    per_frame = defaultdict(RunningStats)
    skipped = []
    if args.skip_flow:
        return {"skipped": True}

    iterator = []
    for c in camera_ids:
        for a, b in zip(frame_ids[:-1], frame_ids[1:]):
            if c not in image_index:
                skipped.append({"camera": c, "frame": a, "reason": "missing_image_camera"})
                continue
            img_a, img_b = resolve_image_pair(image_index[c], frame_ids, a, b)
            if img_a is None or img_b is None:
                skipped.append({"camera": c, "frame": a, "reason": "missing_image_pair"})
                continue
            cache_path = None
            if args.flow_cache_dir:
                cache_path = Path(args.flow_cache_dir) / f"cam_{c:04d}_{a:06d}_to_{b:06d}.npz"
            iterator.append((c, a, b, img_a, img_b, cache_path))

    pbar = tqdm(range(0, len(iterator), max(1, args.eval_batch_frames)), desc="Flow-warped depth", disable=args.no_progress)
    for start in pbar:
        batch_items = iterator[start:start + max(1, args.eval_batch_frames)]
        if not batch_items:
            continue
        flow_items = [(img_a, img_b, cache_path) for _, _, _, img_a, img_b, cache_path in batch_items]
        flow_arrays = get_raft_flows_batch(flow_items, args.raft_model, args.device, args.raft_batch_size)
        if len(flow_arrays) != len(batch_items):
            raise RuntimeError(f"Expected {len(batch_items)} flows, got {len(flow_arrays)}")

        depths_a = []
        depths_b = []
        flows = []
        metas = []
        for (c, a, b, _, _, _), flow in zip(batch_items, flow_arrays):
            depth_a = depth_cache.get(c, a)
            depth_b = depth_cache.get(c, b, target_hw=tuple(depth_a.shape))
            depths_a.append(depth_a)
            depths_b.append(depth_b)
            flows.append(resize_flow_tensor(flow, tuple(depth_a.shape), device))
            metas.append((c, a))

        if len({tuple(d.shape) for d in depths_a}) != 1:
            # Rare mixed-resolution case: split to single-item batches to avoid padding policy changes.
            for (c, a, b, _, _, _), flow in zip(batch_items, flow_arrays):
                depth_a = depth_cache.get(c, a)
                depth_b = depth_cache.get(c, b, target_hw=tuple(depth_a.shape))
                flow_t = resize_flow_tensor(flow, tuple(depth_a.shape), device)
                _accumulate_flow_depth_pair(depth_a, depth_b, flow_t, c, a, overall, per_camera, per_frame, args, device)
            continue

        depth_a_batch = torch.stack(depths_a, dim=0)
        depth_b_batch = torch.stack(depths_b, dim=0)
        flow_batch = torch.stack(flows, dim=0)
        h, w = depth_a_batch.shape[-2:]
        grid = make_grid_torch(h, w, args.flow_sample_stride, device)
        flow_sample, flow_valid = bilinear_sample_torch(flow_batch, grid)
        warped = grid[None] + flow_sample
        target_depth, target_valid = bilinear_sample_torch(depth_b_batch, warped)
        source_depth, source_valid = bilinear_sample_torch(depth_a_batch, grid)
        valid = (
            flow_valid
            & target_valid
            & source_valid
            & torch.isfinite(source_depth)
            & torch.isfinite(target_depth)
            & (source_depth > args.min_depth)
            & (target_depth > args.min_depth)
        )
        if args.max_depth is not None:
            valid &= (source_depth < args.max_depth) & (target_depth < args.max_depth)

        err_all = (source_depth - target_depth).abs()
        for idx, (c, a) in enumerate(metas):
            mask = valid[idx]
            if not torch.any(mask):
                continue
            err = err_all[idx][mask]
            denom = target_depth[idx][mask]
            add_torch_error_stats(overall, err, denom)
            add_torch_error_stats(per_camera[c], err, denom)
            add_torch_error_stats(per_frame[a], err, denom)

    if args.profile_timing:
        print(f"[TIMING] Flow-warped depth: {time.time() - start_time:.3f}s")

    return {
        "overall": overall.as_error_dict(),
        "per_camera": {str(k): v.as_error_dict() for k, v in sorted(per_camera.items())},
        "per_frame": {str(k): v.as_error_dict() for k, v in sorted(per_frame.items())},
        "skipped": skipped,
    }


def _accumulate_flow_depth_pair(depth_a, depth_b, flow, c, a, overall, per_camera, per_frame, args, device):
    h, w = depth_a.shape
    grid = make_grid_torch(h, w, args.flow_sample_stride, device)
    flow_sample, flow_valid = bilinear_sample_torch(flow[None], grid)
    warped = grid + flow_sample[0]
    target_depth, target_valid = bilinear_sample_torch(depth_b, warped)
    source_depth, source_valid = bilinear_sample_torch(depth_a, grid)
    valid = (
        flow_valid[0]
        & target_valid
        & source_valid
        & torch.isfinite(source_depth)
        & torch.isfinite(target_depth)
        & (source_depth > args.min_depth)
        & (target_depth > args.min_depth)
    )
    if args.max_depth is not None:
        valid &= (source_depth < args.max_depth) & (target_depth < args.max_depth)
    if torch.any(valid):
        err = (source_depth[valid] - target_depth[valid]).abs()
        denom = target_depth[valid]
        add_torch_error_stats(overall, err, denom)
        add_torch_error_stats(per_camera[c], err, denom)
        add_torch_error_stats(per_frame[a], err, denom)


def evaluate_flow_depth_cpu(dataset, camera_ids, frame_ids, image_index, args):
    overall = RunningStats()
    per_camera = defaultdict(RunningStats)
    per_frame = defaultdict(RunningStats)
    skipped = []
    if args.skip_flow:
        return {"skipped": True}

    iterator = []
    for c in camera_ids:
        for a, b in zip(frame_ids[:-1], frame_ids[1:]):
            iterator.append((c, a, b))

    for c, a, b in tqdm(iterator, desc="Flow-warped depth", disable=args.no_progress):
        if c not in image_index:
            skipped.append({"camera": c, "frame": a, "reason": "missing_image_camera"})
            continue
        img_a, img_b = resolve_image_pair(image_index[c], frame_ids, a, b)
        if img_a is None or img_b is None:
            skipped.append({"camera": c, "frame": a, "reason": "missing_image_pair"})
            continue

        depth_a = load_depth(
            dataset.records[(c, a)],
            mode=args.depth_resize_mode,
            scale_divisor=args.depth_scale_divisor,
            inverse_depth=args.inverse_depth,
        )
        depth_b = load_depth(
            dataset.records[(c, b)],
            target_hw=depth_a.shape,
            mode=args.depth_resize_mode,
            scale_divisor=args.depth_scale_divisor,
            inverse_depth=args.inverse_depth,
        )
        cache_path = None
        if args.flow_cache_dir:
            cache_path = Path(args.flow_cache_dir) / f"cam_{c:04d}_{a:06d}_to_{b:06d}.npz"
        flow = get_raft_flow(img_a, img_b, cache_path, args.raft_model, args.device)
        flow = resize_flow(flow, depth_a.shape)

        grid = make_grid(depth_a.shape[0], depth_a.shape[1], args.flow_sample_stride)
        flow_sample, flow_valid = bilinear_sample(flow, grid)
        warped = grid + flow_sample
        target_depth, target_valid = bilinear_sample(depth_b, warped)
        source_depth, source_valid = bilinear_sample(depth_a, grid)
        valid = (
            flow_valid
            & target_valid
            & source_valid
            & np.isfinite(source_depth)
            & np.isfinite(target_depth)
            & (source_depth > args.min_depth)
            & (target_depth > args.min_depth)
        )
        if args.max_depth is not None:
            valid &= (source_depth < args.max_depth) & (target_depth < args.max_depth)
        err = np.abs(source_depth[valid] - target_depth[valid])
        denom = target_depth[valid]
        overall.add_errors(err, denom)
        per_camera[c].add_errors(err, denom)
        per_frame[a].add_errors(err, denom)

    return {
        "overall": overall.as_error_dict(),
        "per_camera": {str(k): v.as_error_dict() for k, v in sorted(per_camera.items())},
        "per_frame": {str(k): v.as_error_dict() for k, v in sorted(per_frame.items())},
        "skipped": skipped,
    }


def evaluate_cross_view_and_order(dataset, camera_ids, frame_ids, c2w_all, w2c_all, K_all, pose_map, image_index, args):
    start_time = time.time()
    device = resolve_torch_device(args.device)
    depth_cache = DepthTensorCache(dataset, args, device)
    repro_overall = RunningStats()
    repro_pair = defaultdict(RunningStats)
    repro_camera = defaultdict(RunningStats)
    repro_frame = defaultdict(RunningStats)
    order_overall = RunningStats()
    order_pair = defaultdict(RunningStats)
    order_camera = defaultdict(RunningStats)
    order_frame = defaultdict(RunningStats)

    image_hw_cache = {}
    intrinsic_cache: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    c2w_torch = torch.from_numpy(c2w_all).to(device=device, dtype=torch.float32)
    w2c_torch = torch.from_numpy(w2c_all).to(device=device, dtype=torch.float32)

    def get_image_hw(cam):
        if cam in image_hw_cache:
            return image_hw_cache[cam]
        hw = None
        if cam in image_index and image_index[cam]:
            path = next(iter(image_index[cam].values()))
            with Image.open(path) as img:
                w, h = img.size
            hw = (h, w)
        image_hw_cache[cam] = hw
        return hw

    def get_intrinsics(cam, depth_hw):
        pose_idx = pose_map[cam]
        key = (cam, depth_hw[0], depth_hw[1])
        if key not in intrinsic_cache:
            K = scaled_intrinsic_torch(K_all[pose_idx], depth_hw, get_image_hw(cam), device)
            intrinsic_cache[key] = (K, torch.linalg.inv(K))
        return intrinsic_cache[key]

    for frame_id in tqdm(frame_ids, desc="Cross-view depth/order", disable=args.no_progress):
        depths = {cam: depth_cache.get(cam, frame_id) for cam in camera_ids}
        for src in camera_ids:
            depth_src = depths[src]
            h, w = depth_src.shape
            grid = make_grid_torch(h, w, args.sample_stride, device)
            z_src, src_valid = bilinear_sample_torch(depth_src, grid)
            valid_src = src_valid & torch.isfinite(z_src) & (z_src > args.min_depth)
            if args.max_depth is not None:
                valid_src &= z_src < args.max_depth
            if not torch.any(valid_src):
                continue

            src_pose_idx = pose_map[src]
            K_src, K_inv_src = get_intrinsics(src, (h, w))
            points_world = backproject_torch(grid[valid_src], z_src[valid_src], K_inv_src, c2w_torch[src_pose_idx])

            for tgt in camera_ids:
                if src == tgt:
                    continue
                depth_tgt = depths[tgt]
                tgt_pose_idx = pose_map[tgt]
                K_tgt, _ = get_intrinsics(tgt, tuple(depth_tgt.shape))
                uv_tgt, z_proj = project_torch(points_world, K_tgt, w2c_torch[tgt_pose_idx])
                z_tgt, tgt_valid = bilinear_sample_torch(depth_tgt, uv_tgt)
                valid_tgt = (
                    tgt_valid
                    & torch.isfinite(z_proj)
                    & torch.isfinite(z_tgt)
                    & (z_proj > args.min_depth)
                    & (z_tgt > args.min_depth)
                )
                if args.max_depth is not None:
                    valid_tgt &= (z_proj < args.max_depth) & (z_tgt < args.max_depth)
                if not torch.any(valid_tgt):
                    continue

                z_proj_v = z_proj[valid_tgt]
                z_tgt_v = z_tgt[valid_tgt]
                err = (z_proj_v - z_tgt_v).abs()
                pair_key = f"{src}->{tgt}"
                add_torch_error_stats(repro_overall, err, z_tgt_v)
                add_torch_error_stats(repro_pair[pair_key], err, z_tgt_v)
                add_torch_error_stats(repro_camera[src], err, z_tgt_v)
                add_torch_error_stats(repro_frame[frame_id], err, z_tgt_v)

                margin = args.order_margin + args.order_relative_margin * z_tgt_v.abs()
                penalty = (z_tgt_v - z_proj_v) / torch.clamp(z_tgt_v.abs(), min=EPS)
                violations = z_proj_v < (z_tgt_v - margin)
                threshold_violations = {
                    threshold_key(threshold): z_proj_v < (z_tgt_v - threshold)
                    for threshold in args.occlusion_thresholds
                }
                add_torch_order_stats(order_overall, penalty, violations, threshold_violations)
                add_torch_order_stats(order_pair[pair_key], penalty, violations, threshold_violations)
                add_torch_order_stats(order_camera[src], penalty, violations, threshold_violations)
                add_torch_order_stats(order_frame[frame_id], penalty, violations, threshold_violations)

        depth_cache.clear()

    if args.profile_timing:
        print(f"[TIMING] Cross-view depth/order: {time.time() - start_time:.3f}s")

    return {
        "cross_view_reprojection": {
            "overall": repro_overall.as_error_dict(),
            "overall_pair_mean": mean_metric_dict([v.as_error_dict() for v in repro_pair.values()], ["mean", "rmse", "absrel"]),
            "per_pair": {k: v.as_error_dict() for k, v in sorted(repro_pair.items())},
            "per_camera": {str(k): v.as_error_dict() for k, v in sorted(repro_camera.items())},
            "per_frame": {str(k): v.as_error_dict() for k, v in sorted(repro_frame.items())},
        },
        "depth_order": {
            "overall": order_overall.as_order_dict(),
            "overall_pair_mean": mean_metric_dict(
                [v.as_order_dict() for v in order_pair.values()],
                ["violation_rate", "penalty_mean"]
                + [f"violation_rate@{threshold_key(threshold)}" for threshold in args.occlusion_thresholds],
            ),
            "per_pair": {k: v.as_order_dict() for k, v in sorted(order_pair.items())},
            "per_camera": {str(k): v.as_order_dict() for k, v in sorted(order_camera.items())},
            "per_frame": {str(k): v.as_order_dict() for k, v in sorted(order_frame.items())},
        },
    }


def mean_metric_dict(items: Sequence[dict], keys: Sequence[str]) -> dict:
    out = {"count": len(items)}
    for key in keys:
        vals = [item.get(key) for item in items if item.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def threshold_key(value: float) -> str:
    return f"{value:g}"


def write_summary_csv(results: dict, output_path: Path):
    rows = []
    flow = results.get("flow_warped_depth", {})
    if flow and not flow.get("skipped"):
        rows.append({"metric_group": "flow_warped_depth", "scope": "overall", **flow["overall"]})

    cross = results["cross_view_reprojection"]
    if cross and not cross.get("skipped"):
        rows.append({"metric_group": "cross_view_reprojection", "scope": "overall", **cross["overall"]})
    order = results["depth_order"]
    if order and not order.get("skipped"):
        rows.append({"metric_group": "depth_order", "scope": "overall", **order["overall"]})
    keys = sorted({k for row in rows for k in row})
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate depth consistency for one depth directory.")
    parser.add_argument("--depth_dir", required=True, type=str, help="Depth root to evaluate.")
    parser.add_argument("--image_dir", type=str, default=None, help="RGB image root for RAFT and intrinsic scaling.")
    parser.add_argument("--pose_file", required=True, type=str, help="pose_metadata.json with poses and intrinsics.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory for metrics.json/csv/config.json.")
    parser.add_argument("--layout", choices=["auto", "priors", "render_flat", "render_by_cam"], default="auto")
    parser.add_argument("--camera_ids", type=int, nargs="*", default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--sample_stride", type=int, default=4, help="Pixel stride for cross-view/order metrics.")
    parser.add_argument("--flow_sample_stride", type=int, default=4, help="Pixel stride for flow-warp metric.")

    parser.add_argument("--order_margin", type=float, default=1e-2)
    parser.add_argument("--order_relative_margin", type=float, default=0.0)
    parser.add_argument(
        "--occlusion_thresholds",
        type=float,
        nargs="*",
        default=[0.0, 1e-3, 1e-2, 1e-1],
        help="Absolute depth margins used for extra occlusion/violation rates.",
    )
    parser.add_argument("--min_depth", type=float, default=0.0)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument(
        "--depth_scale_divisor",
        type=float,
        default=1.0,
        help="Divide every loaded depth value by this factor before evaluating.",
    )
    parser.add_argument("--inverse_depth", action="store_true", help="Convert loaded inverse-depth values to depth with 1 / value.")
    parser.add_argument("--depth_resize_mode", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--skip_flow", action="store_true", help="Skip RAFT flow-warped depth metric.")
    parser.add_argument("--skip_cross_view_order", action="store_true", help="Skip cross-view reprojection and depth-order metrics.")

    parser.add_argument("--flow_cache_dir", type=str, default=None)
    parser.add_argument("--raft_model", choices=["small", "large"], default="large")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_batch_frames", type=int, default=8, help="Batch size for Torch depth metric evaluation.")
    parser.add_argument("--raft_batch_size", type=int, default=4, help="Batch size for uncached RAFT optical-flow inference.")
    parser.add_argument("--depth_cache_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--profile_timing", action="store_true", help="Print per-metric evaluation timing.")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = discover_depth_records(Path(args.depth_dir), args.layout)
    c2w_all, w2c_all, K_all, camera_names = load_pose_metadata(Path(args.pose_file))
    camera_ids, frame_ids = select_cameras_and_frames(dataset, args)
    if len(camera_ids) < 1:
        raise ValueError("No cameras selected.")
    if len(frame_ids) < 1:
        raise ValueError("No complete frames selected.")
    if len(camera_ids) < 2:
        print("[WARNING] Fewer than 2 cameras selected; cross-view and depth-order metrics will be empty.")

    pose_map = camera_pose_index(camera_ids, len(c2w_all))
    image_index, image_mapping = build_image_index(Path(args.image_dir) if args.image_dir else None, sorted(camera_ids), camera_names)

    print(f"[INFO] Depth layout: {dataset.layout}")
    print(f"[INFO] Cameras: {camera_ids}")
    print(f"[INFO] Frames: {frame_ids[0]}..{frame_ids[-1]} ({len(frame_ids)} complete frames)")
    print(f"[INFO] Pose mapping: {pose_map}")

    flow_results = evaluate_flow_depth(dataset, camera_ids, frame_ids, image_index, args)
    if args.skip_cross_view_order:
        cross_order = {
            "cross_view_reprojection": {"skipped": True},
            "depth_order": {"skipped": True},
        }
    else:
        cross_order = evaluate_cross_view_and_order(
            dataset, camera_ids, frame_ids, c2w_all, w2c_all, K_all, pose_map, image_index, args
        )

    results = {
        "flow_warped_depth": flow_results,
        **cross_order,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    write_summary_csv(results, output_dir / "metrics.csv")

    config = {
        "depth_dir": str(Path(args.depth_dir).resolve()),
        "layout": dataset.layout,
        "image_dir": str(Path(args.image_dir).resolve()) if args.image_dir else None,
        "pose_file": str(Path(args.pose_file).resolve()),
        "camera_ids": camera_ids,
        "frame_ids": frame_ids,
        "pose_map": pose_map,
        "image_mapping": image_mapping,
        "args": vars(args),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)

    print(f"[DONE] Wrote {output_dir / 'metrics.json'}")
    print(f"[DONE] Wrote {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
