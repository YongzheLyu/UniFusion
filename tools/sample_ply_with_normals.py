#!/usr/bin/env python3
"""
使用Open3D随机采样PLY文件中的点并计算法向量

这个脚本使用Open3D库从PLY文件中随机采样指定数量的点，并为这些点计算法向量。
支持高效的点云处理和法向量估计。

使用方法:
    python sample_ply_with_normals_open3d.py --input path/to/pointcloud.ply --max_n 1000 --output sampled_points.ply
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional
import sys
import open3d as o3d


def read_ply_file(filepath: Path) -> o3d.geometry.PointCloud:
    """
    使用Open3D读取PLY文件
    
    Args:
        filepath: PLY文件路径
        
    Returns:
        point_cloud: Open3D点云对象
    """
    try:
        point_cloud = o3d.io.read_point_cloud(str(filepath))
        
        if point_cloud.is_empty():
            raise ValueError(f"点云文件为空: {filepath}")
        
        print(f"[INFO] 成功读取PLY文件: {filepath.name}")
        print(f"  点数: {len(point_cloud.points)}")
        print(f"  包含颜色: {point_cloud.has_colors()}")
        print(f"  包含法向量: {point_cloud.has_normals()}")
        
        return point_cloud
        
    except Exception as e:
        print(f"[ERROR] 读取PLY文件失败: {e}")
        raise


def random_sample_points(point_cloud: o3d.geometry.PointCloud, max_n: int, 
                        seed: Optional[int] = None) -> o3d.geometry.PointCloud:
    """
    随机采样点云数据
    
    Args:
        point_cloud: 原始点云
        max_n: 最大采样点数
        seed: 随机种子
        
    Returns:
        sampled_point_cloud: 采样后的点云
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = len(point_cloud.points)
    
    if n_points <= max_n:
        print(f"[INFO] 点数 {n_points} <= 最大采样数 {max_n}，返回所有点")
        return point_cloud
    
    # 随机选择索引
    indices = np.random.choice(n_points, max_n, replace=False)
    
    # 创建新的点云对象
    sampled_point_cloud = o3d.geometry.PointCloud()
    
    # 采样点坐标
    points = np.asarray(point_cloud.points)[indices]
    sampled_point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # 采样颜色（如果存在）
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)[indices]
        sampled_point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # 采样法向量（如果存在）
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)[indices]
        sampled_point_cloud.normals = o3d.utility.Vector3dVector(normals)
    
    print(f"[INFO] 随机采样: {n_points} -> {max_n} 点")
    
    return sampled_point_cloud


def compute_normals_open3d(point_cloud: o3d.geometry.PointCloud, 
                          radius: float = 0.1,
                          max_nn: int = 30) -> o3d.geometry.PointCloud:
    """
    使用Open3D计算点云法向量
    
    Args:
        point_cloud: 输入点云
        radius: 搜索半径
        max_nn: 最大邻居点数
        
    Returns:
        point_cloud_with_normals: 包含法向量的点云
    """
    print(f"[INFO] 使用Open3D计算法向量 (radius={radius}, max_nn={max_nn})...")
    
    # 创建点云的副本以避免修改原始数据
    point_cloud_with_normals = o3d.geometry.PointCloud(point_cloud)
    
    # 估计法向量
    point_cloud_with_normals.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    
    # 确保法向量朝向一致
    point_cloud_with_normals.orient_normals_consistent_tangent_plane(100)
    
    print(f"[INFO] 法向量计算完成")
    
    return point_cloud_with_normals


def save_point_cloud(point_cloud: o3d.geometry.PointCloud, filepath: Path):
    """
    保存点云到文件
    
    Args:
        point_cloud: 要保存的点云
        filepath: 输出文件路径
    """
    try:
        success = o3d.io.write_point_cloud(str(filepath), point_cloud)
        
        if success:
            print(f"[INFO] 点云已保存: {filepath.name}")
            print(f"  点数: {len(point_cloud.points)}")
            print(f"  包含颜色: {point_cloud.has_colors()}")
            print(f"  包含法向量: {point_cloud.has_normals()}")
        else:
            print(f"[ERROR] 保存点云失败: {filepath}")
            
    except Exception as e:
        print(f"[ERROR] 保存点云时出错: {e}")


def visualize_point_cloud(point_cloud: o3d.geometry.PointCloud, 
                         window_name: str = "Point Cloud"):
    """
    可视化点云
    
    Args:
        point_cloud: 要可视化的点云
        window_name: 窗口名称
    """
    print(f"[INFO] 正在可视化点云...")
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # 添加点云
    vis.add_geometry(point_cloud)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    
    # 如果有点法向量，可以显示
    if point_cloud.has_normals():
        render_option.show_coordinate_frame = True
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def process_point_cloud(input_path: Path, max_n: int, output_path: Optional[Path] = None,
                       radius: float = 0.1, max_nn: int = 30, seed: Optional[int] = None,
                       visualize: bool = False):
    """
    处理点云：读取、采样、计算法向量、保存
    
    Args:
        input_path: 输入PLY文件路径
        max_n: 最大采样点数
        output_path: 输出PLY文件路径
        radius: 法向量计算搜索半径
        max_nn: 最大邻居点数
        seed: 随机种子
        visualize: 是否可视化结果
    """
    print("=" * 60)
    print("[INFO] 开始处理点云...")
    print(f"  输入文件: {input_path}")
    print(f"  最大采样数: {max_n}")
    print(f"  输出文件: {output_path}")
    print("=" * 60)
    
    # 1. 读取点云
    print("\n[STEP 1] 读取点云...")
    point_cloud = read_ply_file(input_path)
    
    # 2. 随机采样
    print("\n[STEP 2] 随机采样...")
    sampled_point_cloud = random_sample_points(point_cloud, max_n, seed)
    
    # 3. 计算法向量
    print("\n[STEP 3] 计算法向量...")
    if not sampled_point_cloud.has_normals():
        sampled_point_cloud = compute_normals_open3d(sampled_point_cloud, radius, max_nn)
    else:
        print("[INFO] 点云已包含法向量，跳过计算")
    
    # 4. 保存结果
    if output_path:
        print("\n[STEP 4] 保存结果...")
        save_point_cloud(sampled_point_cloud, output_path)
    
    # 5. 可视化（可选）
    if visualize:
        print("\n[STEP 5] 可视化...")
        visualize_point_cloud(sampled_point_cloud, "Sampled Point Cloud with Normals")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 处理完成！")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用Open3D随机采样PLY文件中的点并计算法向量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法 - 采样1000个点并计算法向量
    python sample_ply_with_normals_open3d.py --input pointcloud.ply --max_n 1000
    
    # 指定输出文件和参数
    python sample_ply_with_normals_open3d.py --input pointcloud.ply --max_n 1000 --output sampled.ply --radius 0.05 --max_nn 20
    
    # 包含可视化
    python sample_ply_with_normals_open3d.py --input pointcloud.ply --max_n 1000 --visualize
    
    # 设置随机种子
    python sample_ply_with_normals_open3d.py --input pointcloud.ply --max_n 1000 --seed 42
        """
    )
    
    parser.add_argument("--input", "-i", type=Path, required=True,
                       help="输入PLY文件路径")
    
    parser.add_argument("--max_n", "-n", type=int, required=True,
                       help="最大采样点数")
    
    parser.add_argument("--output", "-o", type=Path, 
                       help="输出PLY文件路径 (默认: sampled_<原文件名>)")
    
    parser.add_argument("--radius", "-r", type=float, default=0.1,
                       help="法向量计算搜索半径 (默认: 0.1)")
    
    parser.add_argument("--max_nn", type=int, default=30,
                       help="最大邻居点数 (默认: 30)")
    
    parser.add_argument("--seed", "-s", type=int,
                       help="随机种子")
    
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="可视化结果")
    
    parser.add_argument("--dry_run", action="store_true",
                       help="预览操作，不实际处理")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查输入文件
    if not args.input.exists():
        print(f"[ERROR] 输入文件不存在: {args.input}")
        return 1
    
    if not args.input.suffix.lower() == '.ply':
        print(f"[ERROR] 输入文件必须是PLY格式: {args.input}")
        return 1
    
    # 设置默认输出文件名
    if not args.output:
        args.output = args.input.parent / f"sampled_{args.max_n}_{args.input.name}"
    
    # 预览模式
    if args.dry_run:
        print("[DRY RUN] 预览操作:")
        print(f"  输入文件: {args.input}")
        print(f"  采样点数: {args.max_n}")
        print(f"  输出文件: {args.output}")
        print(f"  搜索半径: {args.radius}")
        print(f"  最大邻居数: {args.max_nn}")
        print(f"  随机种子: {args.seed}")
        print(f"  可视化: {args.visualize}")
        return 0
    
    try:
        # 执行处理
        process_point_cloud(
            input_path=args.input,
            max_n=args.max_n,
            output_path=args.output,
            radius=args.radius,
            max_nn=args.max_nn,
            seed=args.seed,
            visualize=args.visualize
        )
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
