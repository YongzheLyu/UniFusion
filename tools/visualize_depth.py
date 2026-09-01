import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def collect_depth_files(input_paths):
    """从输入路径中收集所有深度文件"""
    depth_files = []
    for path in input_paths:
        path = Path(path)
        if path.is_file():
            if path.suffix == '.npy':
                depth_files.append(path)
        elif path.is_dir():
            # 递归查找目录中的所有 .npy 文件
            depth_files.extend(sorted(path.rglob('*.npy')))
    return depth_files


def compute_global_min_max(depth_files):
    """计算所有深度文件的全局 min 和 max"""
    global_min = float('inf')
    global_max = float('-inf')

    print(f"Computing global min/max from {len(depth_files)} files...")
    for depth_path in depth_files:
        depth = np.load(depth_path)
        global_min = min(global_min, depth.min())
        global_max = max(global_max, depth.max())
        print(f"  {depth_path.name}: min={depth.min():.6f}, max={depth.max():.6f}")

    print(f"\nGlobal range: [{global_min:.6f}, {global_max:.6f}]")
    return global_min, global_max


def visualize_depth(depth_path, output_dir=None, global_min=None, global_max=None, simple_mode=False):
    """可视化深度图"""
    # 加载深度数据
    depth = np.load(depth_path)

    print(f"\nDepth file: {depth_path}")
    print(f"Shape: {depth.shape}")
    print(f"Dtype: {depth.dtype}")
    print(f"Min: {depth.min():.6f}, Max: {depth.max():.6f}, Mean: {depth.mean():.6f}")

    # 确定归一化范围
    if global_min is not None and global_max is not None:
        vmin, vmax = global_min, global_max
        norm_info = f" (global norm: [{vmin:.4f}, {vmax:.4f}])"
    else:
        vmin, vmax = depth.min(), depth.max()
        norm_info = " (local norm)"

    # 简洁模式：只使用一种热力图，无文字和range
    if simple_mode:
        fig, ax = plt.subplots(figsize=(depth.shape[1] / 100, depth.shape[0] / 100), dpi=100)
        ax.imshow(depth, cmap='jet', vmin=vmin, vmax=vmax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Depth Visualization: {Path(depth_path).name}{norm_info}', fontsize=14)

        # 1. 原始深度（使用全局归一化）
        depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
        axes[0, 0].imshow(depth_normalized, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Raw Depth (normalized)')
        axes[0, 0].axis('off')

        # 2. 伪彩色深度（使用全局 vmin/vmax）
        im = axes[0, 1].imshow(depth, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Pseudo-color Depth')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 3. 热力图（使用全局 vmin/vmax）
        im = axes[0, 2].imshow(depth, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('Heatmap')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # 4. Turbo colormap（使用全局 vmin/vmax）
        im = axes[1, 0].imshow(depth, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Turbo Colormap')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 5. 对数尺度（如果深度范围很大）
        if depth.min() > 0:
            depth_log = np.log(depth)
            log_vmin, log_vmax = np.log(vmin) if vmin > 0 else depth_log.min(), np.log(vmax)
            im = axes[1, 1].imshow(depth_log, cmap='viridis', vmin=log_vmin, vmax=log_vmax)
            axes[1, 1].set_title('Log Scale')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        else:
            axes[1, 1].text(0.5, 0.5, 'Log scale not available\n(depth has non-positive values)',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')

        # 6. 深度直方图（显示全局范围）
        axes[1, 2].hist(depth.flatten(), bins=50, edgecolor='black')
        axes[1, 2].axvline(vmin, color='red', linestyle='--', label=f'norm min: {vmin:.4f}')
        axes[1, 2].axvline(vmax, color='red', linestyle='--', label=f'norm max: {vmax:.4f}')
        axes[1, 2].set_title('Depth Histogram')
        axes[1, 2].set_xlabel('Depth Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        if global_min is not None:
            axes[1, 2].legend(fontsize=8)

        plt.tight_layout()

    # 保存或显示
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"{Path(depth_path).stem}_viz.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0 if simple_mode else 0.1)
        print(f"Saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize depth.npy files')
    parser.add_argument('input_paths', nargs='+', type=str,
                        help='Path(s) to depth.npy file(s) or directory(s) containing .npy files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (if not specified, display interactively)')
    parser.add_argument('--no-global-norm', action='store_true',
                        help='Disable global normalization (use per-file normalization)')
    parser.add_argument('--simple', '-s', action='store_true',
                        help='Simple mode: only use jet heatmap without range and text')
    args = parser.parse_args()

    # 收集所有深度文件
    depth_files = collect_depth_files(args.input_paths)

    if not depth_files:
        print("No .npy files found in the specified paths!")
        return

    print(f"Found {len(depth_files)} depth file(s) to visualize")

    # 计算全局归一化范围（除非禁用）
    global_min, global_max = None, None
    if len(depth_files) > 1 and not args.no_global_norm:
        global_min, global_max = compute_global_min_max(depth_files)
    elif args.no_global_norm:
        print("Using per-file normalization (global normalization disabled)")

    # 可视化每个文件
    for depth_path in depth_files:
        visualize_depth(depth_path, args.output, global_min, global_max, args.simple)


if __name__ == '__main__':
    main()
