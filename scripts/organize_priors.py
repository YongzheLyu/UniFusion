#!/usr/bin/env python3
"""
组织先验数据工具

将 all_people_visualization 下的 depth、conf 文件和 charts 目录移动到 priors 文件夹，
并按照 cam_{四位镜头编号}_{四位帧号} 的格式重命名文件。
charts 目录会按照 frame_{帧号} 的格式组织。

使用方法:
    python organize_priors.py --input_folder path/to/all_people_visualization
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Optional
import os
import re
from tqdm import tqdm


def natural_sort_key(filename: str) -> List[object]:
    """Return a key list so that Path objects are sorted in natural order."""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", filename)]


def extract_camera_and_frame_numbers(filename: str) -> tuple:
    """从文件名中提取相机编号和帧编号"""
    # 尝试匹配不同的命名模式
    patterns = [
        r'cam(\d+).*?(\d{4,})',  # cam00_cam_0001_0001_depth.npy
        r'cam(\d+).*?frame_(\d+)',  # cam00_frame_0001_depth.npy
        r'(\d+).*?(\d{4,})',      # 00_0001_depth.npy
        r'.*?(\d+).*?(\d+)',      # 通用模式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            cam_num = int(match.group(1))
            frame_num = int(match.group(2))
            return cam_num, frame_num
    
    # 如果无法匹配，返回默认值
    print(f"[WARNING] 无法从文件名提取编号: {filename}，使用默认值")
    return 0, 0


def organize_depth_and_conf_files(
    input_folder: Path,
    priors_folder: Optional[Path] = None,
    dry_run: bool = False
):
    """组织 depth 和 conf 文件到 priors 文件夹"""
    
    if not input_folder.exists():
        print(f"[ERROR] 输入文件夹不存在: {input_folder}")
        return
    
    # 设置 priors 文件夹路径
    if priors_folder is None:
        priors_folder = input_folder.parent / "priors"
    
    print(f"[INFO] 输入文件夹: {input_folder}")
    print(f"[INFO] 目标文件夹: {priors_folder}")
    depth_priors_folder = priors_folder / "depths"
    conf_priors_folder = priors_folder / "confs"
    charts_priors_folder = priors_folder / "charts"
    # 创建 priors 文件夹
    if not dry_run:
        priors_folder.mkdir(parents=True, exist_ok=True)
        depth_priors_folder.mkdir(parents=True, exist_ok=True)
        conf_priors_folder.mkdir(parents=True, exist_ok=True)
        charts_priors_folder.mkdir(parents=True, exist_ok=True)
    
    # 获取所有帧目录（自然排序），最终输出帧号按遍历顺序（从1开始），不依赖原文件名中的帧号
    frame_dirs = sorted([d for d in input_folder.iterdir() if d.is_dir() and d.name.startswith("frame_")])
    
    if not frame_dirs:
        print("[ERROR] 没有找到帧目录")
        return
    
    print(f"[INFO] 找到 {len(frame_dirs)} 个帧目录")
    
    # 收集所有需要移动的文件
    files_to_move = []
    charts_dirs_to_move = []
    
    for idx, frame_dir in enumerate(frame_dirs):
        frame_name = frame_dir.name
        # 统一使用遍历顺序作为输出帧号（第一帧为 1），避免依赖原始文件名中的帧号
        frame_num = idx + 1
        
        # 处理 depth 文件夹
        depth_dir = frame_dir / "depth"
        conf_dir = frame_dir / "confidence"
        charts_dir = frame_dir / "charts"
        
        if depth_dir.exists():
            # 遍历所有 depth / conf 文件，输出帧号一律用 frame_num（遍历顺序），忽略文件名内嵌帧号
            depth_files = sorted(depth_dir.glob("*.npy"), key=lambda p: natural_sort_key(p.name))
            for depth_file in depth_files:
                cam_num, _ = extract_camera_and_frame_numbers(depth_file.name)
                new_name = f"cam_{cam_num:04d}_{frame_num:04d}.npy"
                target_path = depth_priors_folder / new_name
                files_to_move.append((depth_file, target_path, "depth"))

            if conf_dir.exists():
                conf_files = sorted(conf_dir.glob("*.npy"), key=lambda p: natural_sort_key(p.name))
                for conf_file in conf_files:
                    cam_num, _ = extract_camera_and_frame_numbers(conf_file.name)
                    new_name = f"cam_{cam_num:04d}_{frame_num:04d}.npy"
                    target_path = conf_priors_folder / new_name
                    files_to_move.append((conf_file, target_path, "conf"))
        
        # 处理 charts 文件夹（同样使用遍历顺序帧号）
        if charts_dir.exists() and charts_dir.is_dir():
            target_charts_dir = charts_priors_folder / f"frame_{frame_num:05d}"
            charts_dirs_to_move.append((charts_dir, target_charts_dir, frame_num))
    
    print(f"[INFO] 找到 {len(files_to_move)} 个文件需要移动")
    print(f"[INFO] 找到 {len(charts_dirs_to_move)} 个 charts 目录需要移动")
    
    if dry_run:
        print("\n[DRY RUN] 预览将要执行的操作:")
        for src_file, target_path, file_type in files_to_move:
            print(f"  移动: {src_file} -> {target_path} ({file_type})")
        for src_charts_dir, target_charts_dir, frame_num in charts_dirs_to_move:
            print(f"  移动 charts 目录: {src_charts_dir} -> {target_charts_dir} (frame {frame_num})")
        return
    
    # 执行文件移动
    moved_count = 0
    for src_file, target_path, file_type in tqdm(files_to_move, desc="移动文件"):
        try:
            if src_file.exists():
                shutil.copy(str(src_file), str(target_path))
                moved_count += 1
                print(f"[INFO] 已移动: {src_file.name} -> {target_path.name}")
            else:
                print(f"[WARNING] 源文件不存在: {src_file}")
        except Exception as e:
            print(f"[ERROR] 移动文件失败: {src_file} -> {target_path}: {e}")
    
    # 执行 charts 目录移动
    charts_moved_count = 0
    for src_charts_dir, target_charts_dir, frame_num in tqdm(charts_dirs_to_move, desc="移动 charts 目录"):
        try:
            if src_charts_dir.exists():
                # 如果目标目录已存在，先删除
                if target_charts_dir.exists():
                    shutil.rmtree(str(target_charts_dir))
                # 复制整个 charts 目录
                shutil.copytree(str(src_charts_dir), str(target_charts_dir))
                charts_moved_count += 1
                print(f"[INFO] 已移动 charts 目录: {src_charts_dir.name} -> {target_charts_dir.name} (frame {frame_num})")
            else:
                print(f"[WARNING] 源 charts 目录不存在: {src_charts_dir}")
        except Exception as e:
            print(f"[ERROR] 移动 charts 目录失败: {src_charts_dir} -> {target_charts_dir}: {e}")
    
    print(f"\n[SUCCESS] 完成！共移动了 {moved_count} 个文件")
    print(f"[SUCCESS] 共移动了 {charts_moved_count} 个 charts 目录")
    print(f"[INFO] 文件已组织到: {priors_folder}")
    
    # 显示结果统计
    depth_files = list(depth_priors_folder.glob("*.npy"))
    conf_files = list(conf_priors_folder.glob("*.npy"))
    charts_dirs = list(charts_priors_folder.glob("frame_*")) if charts_priors_folder.exists() else []
    
    print(f"[INFO] 深度文件: {len(depth_files)} 个")
    print(f"[INFO] 置信度文件: {len(conf_files)} 个")
    print(f"[INFO] charts 目录: {len(charts_dirs)} 个")
    
    # 显示一些示例文件名
    if depth_files:
        print(f"[INFO] 示例深度文件名: {depth_files[0].name}")
    if conf_files:
        print(f"[INFO] 示例置信度文件名: {conf_files[0].name}")
    if charts_dirs:
        print(f"[INFO] 示例 charts 目录名: {charts_dirs[0].name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="组织先验数据工具 - 将 depth、conf 文件和 charts 目录移动到 priors 文件夹并重命名"
    )
    
    parser.add_argument("input_folder", type=Path, 
                       help="输入文件夹路径 (包含frame_00000, frame_00001等子目录)")
    
    parser.add_argument("--priors-folder", type=Path, default=None,
                       help="目标 priors 文件夹路径 (默认: 输入文件夹的同级目录)")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="预览操作，不实际移动文件")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件夹
    if not args.input_folder.exists():
        print(f"[ERROR] 输入文件夹不存在: {args.input_folder}")
        return
    
    # 执行组织操作
    organize_depth_and_conf_files(
        input_folder=args.input_folder,
        priors_folder=args.priors_folder,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
