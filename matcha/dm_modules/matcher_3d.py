import torch
import numpy as np
import matplotlib.pyplot as plt
from matcha.dm_scene.cameras import CamerasWrapper, P3DCameras
from matcha.dm_utils.rendering import depths_to_points_parallel


def get_points_depth_in_depthmap_parallel(
    pts:torch.Tensor, 
    depthmap:torch.Tensor, 
    cameras:CamerasWrapper,
    padding_mode='zeros',  # 'reflection', 'border'
    znear=1e-6,
):
    """_summary_

    Args:
        pts (torch.Tensor): Has shape (n_depths, N, 3).
        depthmap (torch.Tensor): Has shape (n_depths, H, W) or (n_depths, H, W, 1).
        p3d_camera (P3DCameras): Should contain n_depths cameras.

    Returns:
        _type_: _description_
    """
    n_depths, image_height, image_width = depthmap.shape[:3]

    pts_projections = cameras.transform_points_world_to_view(pts)  # (n_depths, N, 3)
    fov_mask = pts_projections[..., 2] > 0.  # (n_depths, N)
    pts_projections.clamp(min=torch.tensor([[[-1e8, -1e8, znear]]]).to(pts_projections.device))
    
    pts_projections = cameras.project_points(pts_projections, points_are_already_in_view_space=True, znear=znear)  # (n_depths, N, 2)
    fov_mask = fov_mask & pts_projections.isfinite().all(dim=-1)  # (n_depths, N)
    pts_projections = pts_projections.nan_to_num(nan=0., posinf=0., neginf=0.)
    
    if False:
        print("TOREMOVE-pts_projections:", pts_projections.shape)
        print("TOREMOVE-pts_projections Min/Max/Mean/Std:", pts_projections.min(), pts_projections.max(), pts_projections.mean(), pts_projections.std())
        
    factor = -1 * min(image_height, image_width)
    factors = torch.tensor([[[factor / image_width, factor / image_height]]]).to(pts.device)  # (1, 1, 2)
    # pts_projections[..., 0] = factor / image_width * pts_projections[..., 0]
    # pts_projections[..., 1] = factor / image_height * pts_projections[..., 1]
    pts_projections = pts_projections[..., :2] * factors  # (n_depths, N, 2)
    pts_projections = pts_projections.view(n_depths, -1, 1, 2)

    depth_view = depthmap.reshape(n_depths, 1, image_height, image_width)  # (n_depths, 1, H, W)
    map_z = torch.nn.functional.grid_sample(
        input=depth_view,
        grid=pts_projections,
        mode='bilinear',
        padding_mode=padding_mode,  # 'reflection', 'zeros'
        align_corners=False,
    )  # (n_depths, 1, N, 1)
    map_z = map_z[:, 0, :, 0]  # (n_depths, N)
    fov_mask = (map_z > 0.) & fov_mask
    map_z = map_z * fov_mask
    
    return map_z, fov_mask


class Matcher3D:
    def __init__(
        self, 
        cameras:CamerasWrapper,
        reference_pts:torch.Tensor=None, 
        reference_depths:torch.Tensor=None,
    ):
        """_summary_

        Args:
            reference_pts (torch.Tensor): Should have shape (n_charts, height, width, 3).
            reference_depths (torch.Tensor): Should have shape (n_charts, height, width).
            camera (CamerasWrapper): _description_
            match_thr (float): _description_
        """
        self.cameras = cameras
        self.znear = 1e-6
        self.update_references(reference_pts, reference_depths)
        
    @torch.no_grad()
    def update_references(
        self, 
        reference_pts:torch.Tensor=None, 
        reference_depths:torch.Tensor=None,
    ):
        if reference_pts is None and reference_depths is None:
            raise ValueError("Either reference_pts or reference_depths should be provided.")
        
        if reference_depths is None:  
            reference_depths = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(
                reference_pts
            )[..., 2]  # (n_charts, height, width)
            
        if reference_pts is None:
            reference_pts = depths_to_points_parallel(
                reference_depths,
                cameras=self.cameras,
            ).view(*reference_depths.shape, 3)  # (n_charts, height, width, 3)
            
        self.reference_pts = reference_pts  # (n_charts, height, width, 3)
        self.reference_depths = reference_depths  # (n_charts, height, width)
        self.n_charts, self.height, self.width, _ = reference_pts.shape
        self.reference_pts = reference_pts
        
    @torch.no_grad()
    def match(
        self, 
        matching_thr:float, 
        normal_threshold=None
    ):
        if normal_threshold is not None:
            raise NotImplementedError("Normal threshold not implemented yet.")
        
        n_pts_per_chart = self.height * self.width
        points_to_match = self.reference_pts.view(1, -1, 3)  # (1, n_charts * n_pts_per_chart, 3)
        points_to_match = points_to_match.repeat(self.n_charts, 1, 1)  # (n_charts, n_charts * n_pts_per_chart, 3)
        
        # For each camera, get the depth of all points in the camera's view
        true_depth = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(points_to_match)[..., 2]  # (n_charts, n_charts * n_pts_per_chart)
        
        # For each camera, get the depth of the projections of all points in the camera's depth map
        projected_depths, fov_mask = get_points_depth_in_depthmap_parallel(
            pts=points_to_match,  # (n_charts, n_charts * n_pts_per_chart, 3)
            depthmap=self.reference_depths,  # (n_charts, height, width)
            cameras=self.cameras,
            padding_mode='zeros',  # 'reflection', 'border'
            znear=self.znear,
        )  # (n_charts, n_charts * n_pts_per_chart)
        
        # A point is considered a match if the difference between the true depth and the projected depth is low
        depth_errors = (true_depth - projected_depths).abs()
        depth_errors[~fov_mask] = 1e8
        depth_errors = depth_errors.view(self.n_charts, self.n_charts, self.height, self.width)
        
        self.reference_errors = depth_errors
        self.reference_matches = depth_errors < matching_thr

    def visualize_matches_per_chart(self, chart_idx, save_path=None, figsize=(15, 10)):
        """
        可视化指定图表与其他所有图表的匹配情况

        Args:
            chart_idx (int): 要可视化的图表索引
            save_path (str, optional): 保存路径
            figsize (tuple): 图像尺寸
        """
        if not hasattr(self, 'reference_matches'):
            raise ValueError("请先调用 match() 方法")

        n_charts = self.n_charts
        fig, axes = plt.subplots(2, (n_charts + 1) // 2, figsize=figsize)
        axes = axes.flatten() if n_charts > 1 else [axes]

        for target_chart_idx in range(n_charts):
            ax = axes[target_chart_idx]

            # 获取匹配结果
            matches = self.reference_matches[chart_idx, target_chart_idx].cpu().numpy()

            # 可视化匹配结果
            ax.imshow(matches, cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title(f'Chart {chart_idx} → Chart {target_chart_idx}\n'
                        f'Matches: {matches.sum():.0f}/{matches.size} '
                        f'({100*matches.mean():.1f}%)')
            ax.axis('off')

        # 隐藏多余的子图
        for i in range(n_charts, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Matching Results for Chart {chart_idx} with All Other Charts',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        plt.show()

    def visualize_consensus_matches(self, min_consensus=2, save_path=None, figsize=(12, 8)):
        """
        可视化所有图表间一致匹配的像素（共识匹配）

        Args:
            min_consensus (int): 最少需要多少个图表一致匹配
            save_path (str, optional): 保存路径
            figsize (tuple): 图像尺寸
        """
        if not hasattr(self, 'reference_matches'):
            raise ValueError("请先调用 match() 方法")

        n_charts = self.n_charts
        fig, axes = plt.subplots(2, n_charts, figsize=figsize)

        # 计算每个像素被多少个图表匹配
        consensus_count = torch.zeros((n_charts, self.height, self.width), device=self.reference_matches.device)

        for i in range(n_charts):
            # 对于图表i，计算有多少其他图表与它匹配
            matches_for_i = self.reference_matches[i].sum(dim=0)  # (n_charts, height, width) -> (height, width)
            consensus_count[i] = matches_for_i

        consensus_count = consensus_count.cpu().numpy()

        for chart_idx in range(n_charts):
            # 原始共识匹配
            consensus_matches = consensus_count[chart_idx] >= min_consensus

            # 上方：共识匹配可视化
            ax_top = axes[0, chart_idx]
            ax_top.imshow(consensus_matches, cmap='Blues', vmin=0, vmax=1)
            ax_top.set_title(f'Chart {chart_idx} Consensus\n'
                           f'(≥{min_consensus} matches)\n'
                           f'Pixels: {consensus_matches.sum()}/{consensus_matches.size}')
            ax_top.axis('off')

            # 下方：匹配数量热力图
            ax_bottom = axes[1, chart_idx]
            im = ax_bottom.imshow(consensus_count[chart_idx], cmap='viridis', vmin=0, vmax=n_charts-1)
            ax_bottom.set_title(f'Chart {chart_idx} Match Count')
            ax_bottom.axis('off')

            # 添加颜色条
            if chart_idx == n_charts - 1:
                cbar = plt.colorbar(im, ax=ax_bottom, shrink=0.8)
                cbar.set_label('Number of matching charts')

        plt.suptitle(f'Consensus Matching Analysis (Min Consensus: {min_consensus})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved consensus visualization to: {save_path}")

        plt.show()

    def visualize_matching_statistics(self, save_path=None, figsize=(15, 5)):
        """
        可视化匹配统计信息

        Args:
            save_path (str, optional): 保存路径
            figsize (tuple): 图像尺寸
        """
        if not hasattr(self, 'reference_matches'):
            raise ValueError("请先调用 match() 方法")

        n_charts = self.n_charts
        matches = self.reference_matches.cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. 匹配率矩阵
        match_rates = matches.mean(axis=(2, 3))  # (n_charts, n_charts)
        im1 = axes[0].imshow(match_rates, cmap='YlOrRd', vmin=0, vmax=1)
        axes[0].set_title('Match Rates Between Charts')
        axes[0].set_xlabel('Target Chart')
        axes[0].set_ylabel('Source Chart')
        plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Match Rate')

        # 添加数值标签
        for i in range(n_charts):
            for j in range(n_charts):
                axes[0].text(j, i, f'{match_rates[i, j]:.2f}',
                           ha='center', va='center',
                           color='white' if match_rates[i, j] > 0.5 else 'black',
                           fontweight='bold', fontsize=8)

        # 2. 每个图表的平均匹配数
        avg_matches_per_chart = matches.sum(axis=(1, 2, 3)) / (n_charts - 1)  # 除去自匹配
        axes[1].bar(range(n_charts), avg_matches_per_chart)
        axes[1].set_title('Average Matches per Chart')
        axes[1].set_xlabel('Chart Index')
        axes[1].set_ylabel('Average Matches')
        axes[1].grid(True, alpha=0.3)

        # 3. 匹配分布直方图
        all_match_counts = []
        for i in range(n_charts):
            match_count = matches[i].sum(axis=(1, 2))  # 对每个目标图表求和
            all_match_counts.extend(match_count)

        axes[2].hist(all_match_counts, bins=range(n_charts+2), alpha=0.7, edgecolor='black')
        axes[2].set_title('Distribution of Match Counts')
        axes[2].set_xlabel('Number of Matching Charts')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('Matching Statistics Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved statistics visualization to: {save_path}")

        plt.show()

        # 打印详细统计信息
        print("\n" + "="*60)
        print("MATCHING STATISTICS SUMMARY")
        print("="*60)
        print(f"Total Charts: {n_charts}")
        print(f"Image Resolution: {self.width} x {self.height}")
        print(f"Total Pixels per Chart: {self.width * self.height}")

        print("\nMatch Rates Matrix:")
        print("Source→Target ", end="")
        for j in range(n_charts):
            print(f"{j:>8}", end="")
        print()

        for i in range(n_charts):
            print(f"Chart {i}:     ", end="")
            for j in range(n_charts):
                print(f"{match_rates[i, j]:>8.3f}", end="")
            print()

        print("\nAverage matches per chart:")
        for i in range(n_charts):
            print(f"  Chart {i}: {avg_matches_per_chart[i]:.1f}")

        print(f"\nOverall average match rate: {match_rates.mean():.4f}")
        print("="*60)

    def visualize_all(self, output_dir="./matcher_visualizations", min_consensus=2):
        """
        生成所有类型的可视化

        Args:
            output_dir (str): 输出目录
            min_consensus (int): 共识匹配的最小阈值
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Generating matching visualizations...")

        # 1. 为每个图表生成匹配可视化
        for chart_idx in range(self.n_charts):
            self.visualize_matches_per_chart(
                chart_idx,
                save_path=os.path.join(output_dir, f'matches_chart_{chart_idx}.png')
            )
            plt.close()

        # 2. 生成共识匹配可视化
        self.visualize_consensus_matches(
            min_consensus=min_consensus,
            save_path=os.path.join(output_dir, 'consensus_matches.png')
        )
        plt.close()

        # 3. 生成统计信息可视化
        self.visualize_matching_statistics(
            save_path=os.path.join(output_dir, 'matching_statistics.png')
        )
        plt.close()

        print(f"All visualizations saved to: {output_dir}")
    
    def compute_reprojection_errors(
        self, 
        depths=None,
        points=None,
    ):
        """_summary_

        Args:
            depths (_type_, optional): Shape (n_charts, height, width). Defaults to None.
            points (_type_, optional): Shape (n_charts, height, width, 3). Defaults to None.

        Raises:
            ValueError: _description_
        """
        if points is None and depths is None:
            raise ValueError("Either depths or points should be provided.")
        
        if points is None:
            points_to_match = depths_to_points_parallel(
                depths, 
                cameras=self.cameras,
            )  # (n_charts, height, width, 3)
            depths_to_match = depths  # (n_charts, height, width)
        
        if depths is None:
            points_to_match = points  # (n_charts, height, width, 3)
            depths_to_match = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(
                points
            )[..., 2]  # (n_charts, height, width)
        
        n_pts_per_chart = self.height * self.width
        points_to_match = points_to_match.view(1, -1, 3)  # (1, n_charts * n_pts_per_chart, 3)
        points_to_match = points_to_match.repeat(self.n_charts, 1, 1)  # (n_charts, n_charts * n_pts_per_chart, 3)
        
        # For each camera, get the depth of all points in the camera's view
        true_depth = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(points_to_match)[..., 2]  # (n_charts, n_charts * n_pts_per_chart)
        
        # For each camera, get the depth of the projections of all points in the camera's depth map
        projected_depths, fov_mask = get_points_depth_in_depthmap_parallel(
            pts=points_to_match,  # (n_charts, n_charts * n_pts_per_chart, 3)
            depthmap=depths_to_match,  # (n_charts, height, width)
            cameras=self.cameras,
            padding_mode='zeros',  # 'reflection', 'border'
            znear=self.znear,
        )  # (n_charts, n_charts * n_pts_per_chart)
        
        # A point is considered a match if the difference between the true depth and the projected depth is low
        depth_errors = (true_depth - projected_depths).abs().nan_to_num()
        # depth_errors[~fov_mask] = 1e8
        depth_errors = depth_errors.view(self.n_charts, self.n_charts, self.height, self.width)
        fov_mask = fov_mask.view(self.n_charts, self.n_charts, self.height, self.width)
        return depth_errors, fov_mask
