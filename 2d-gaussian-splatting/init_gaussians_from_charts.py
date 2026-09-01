
"""
从 charts_data.npz 初始化高斯并保存为 PLY 文件的独立程序

使用方法:
python init_gaussians_from_charts.py --charts_path /path/to/charts_data.npz --output_path /path/to/output.ply

可选参数:
--n_max_gaussians: 最大高斯数量 (默认: 200000)
--conf_th: 置信度阈值 (默认: -1.0)
--ratio_th: 比率阈值 (默认: 5.0)
--normal_scale: 法向量尺度 (默认: 1e-10)
--normalized_scales: 归一化尺度 (默认: 0.5)
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 导入必要的模块
from scene import GaussianModel
from matcha.dm_scene.charts import load_charts_data, get_gaussian_parameters_from_charts_data
from arguments import ModelParams


def create_dummy_dataset(source_path, model_path):
    """创建一个虚拟的dataset对象，用于初始化GaussianModel"""
    class DummyDataset:
        def __init__(self):
            self.sh_degree = 3  # 默认SH度数
            self.source_path = source_path
            self.model_path = model_path

    return DummyDataset()


def init_gaussians_from_charts(
    charts_data_path,
    output_ply_path,
    n_max_gaussians=200000,
    conf_th=-1.0,
    ratio_th=5.0,
    normal_scale=1e-10,
    normalized_scales=0.5,
    spatial_lr_scale=1.0
):
    """
    从charts_data初始化高斯并保存为PLY文件

    Args:
        charts_data_path: charts_data.npz 文件路径
        output_ply_path: 输出PLY文件路径
        n_max_gaussians: 最大高斯数量
        conf_th: 置信度阈值
        ratio_th: 比率阈值
        normal_scale: 法向量尺度
        normalized_scales: 归一化尺度
        spatial_lr_scale: 空间学习率尺度
    """

    print(f"[INFO] Loading charts data from: {charts_data_path}")
    charts_data = load_charts_data(charts_data_path)

    print(f"[INFO] Charts data shapes:")
    for key, value in charts_data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")

    # 创建虚拟的images (使用charts_data中的默认值)
    # 注意：这里我们使用空列表，因为get_gaussian_parameters_from_charts_data可能不需要实际的图像数据
    _images = []

    print("[INFO] Extracting Gaussian parameters from charts data...")
    gaussian_params = get_gaussian_parameters_from_charts_data(
        charts_data=charts_data,
        images=_images,
        conf_th=conf_th,
        ratio_th=ratio_th,
        normal_scale=normal_scale,
        normalized_scales=normalized_scales,
    )

    print(f"[INFO] Total Gaussians before sampling: {len(gaussian_params['means'])}")

    # 下采样高斯数量（如果需要）
    if n_max_gaussians > 0 and n_max_gaussians < len(gaussian_params['means']):
        downsample_factor = len(gaussian_params['means']) / n_max_gaussians
        sample_idx = torch.randperm(len(gaussian_params['means']))[:n_max_gaussians]
        print(f"[INFO] Downsampling to {n_max_gaussians} Gaussians (factor: {downsample_factor:.2f})")
    else:
        downsample_factor = 1.0
        sample_idx = torch.arange(len(gaussian_params['means']))
        print("[INFO] Using all Gaussians (no downsampling)")

    # 提取参数
    _means = gaussian_params['means'][sample_idx]
    _scales = gaussian_params['scales'][..., :2][sample_idx] * downsample_factor
    _quaternions = gaussian_params['quaternions'][sample_idx]
    _colors = gaussian_params['colors'][sample_idx]

    print(f"[INFO] Parameter shapes:")
    print(f"  means: {_means.shape}")
    print(f"  scales: {_scales.shape}")
    print(f"  quaternions: {_quaternions.shape}")
    print(f"  colors: {_colors.shape}")

    print(f"[INFO] Parameter ranges:")
    print(f"  scales - min: {_scales.min().item():.6f}, max: {_scales.max().item():.6f}")
    print(f"  means - min: {_means.min().item():.6f}, max: {_means.max().item():.6f}")

    # 创建GaussianModel
    print("[INFO] Creating Gaussian model...")
    dataset = create_dummy_dataset(
        source_path=os.path.dirname(charts_data_path),
        model_path=os.path.dirname(output_ply_path)
    )
    gaussians = GaussianModel(dataset.sh_degree, dataset)

    # 初始化高斯参数
    print("[INFO] Initializing Gaussians from parameters...")
    gaussians.create_from_parameters(_means, _scales, _quaternions, _colors, spatial_lr_scale)

    # 保存PLY文件
    print(f"[INFO] Saving PLY file to: {output_ply_path}")
    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    gaussians.save_ply(output_ply_path)

    print(f"[SUCCESS] Saved {len(_means)} Gaussians to {output_ply_path}")
    print(f"[INFO] Final spatial LR scale: {gaussians.spatial_lr_scale}")

    return gaussians


def main():
    parser = argparse.ArgumentParser(description="Initialize Gaussians from charts data and save as PLY")
    parser.add_argument('--charts_path', type=str, required=True,
                       help='Path to charts_data.npz file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to output PLY file')
    parser.add_argument('--n_max_gaussians', type=int, default=200000,
                       help='Maximum number of Gaussians (default: 200000)')
    parser.add_argument('--conf_th', type=float, default=-1.0,
                       help='Confidence threshold (default: -1.0)')
    parser.add_argument('--ratio_th', type=float, default=5.0,
                       help='Ratio threshold (default: 5.0)')
    parser.add_argument('--normal_scale', type=float, default=1e-10,
                       help='Normal scale (default: 1e-10)')
    parser.add_argument('--normalized_scales', type=float, default=0.5,
                       help='Normalized scales (default: 0.5)')
    parser.add_argument('--spatial_lr_scale', type=float, default=1.0,
                       help='Spatial learning rate scale (default: 1.0)')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.charts_path):
        print(f"[ERROR] Charts data file not found: {args.charts_path}")
        sys.exit(1)

    try:
        init_gaussians_from_charts(
            charts_data_path=args.charts_path,
            output_ply_path=args.output_path,
            n_max_gaussians=args.n_max_gaussians,
            conf_th=args.conf_th,
            ratio_th=args.ratio_th,
            normal_scale=args.normal_scale,
            normalized_scales=args.normalized_scales,
            spatial_lr_scale=args.spatial_lr_scale
        )
        print("[SUCCESS] Gaussian initialization completed!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Gaussians: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
