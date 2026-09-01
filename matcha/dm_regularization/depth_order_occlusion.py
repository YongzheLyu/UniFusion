import torch

from matcha.dm_modules.matcher_3d import get_points_depth_in_depthmap_parallel
from matcha.dm_utils.rendering import depths_to_points_parallel
from matcha.dm_scene.cameras import CamerasWrapper


def depth_order_occlusion_loss(
    depths: torch.Tensor,
    cameras: CamerasWrapper,
    penalty: float = 1.0,
    loss_type: str = "hinge",
    reduction: str = "mean",
    padding_mode: str = "zeros",
    znear: float = 1e-6,
):
    """
    基于遮挡一致性的深度排序损失（多视角）。

    步骤:
        1) 将每个视角的深度图 (n, H, W) 反投影为 3D 点 (n, H*W, 3)。
        2) 将这些 3D 点投影到所有视角，得到每个视角下的 uv 坐标与采样深度。
        3) 如果点的真实深度（沿目标视角光线）小于目标视角深度图该 uv 处的深度，
           认为该点遮挡了该像素对应的3D点，施加固定惩罚。

    Args:
        depths (torch.Tensor): (n, H, W) 或 (n, H, W, 1) 的深度图。
        cameras (CamerasWrapper): 相机包装，需包含 n 个相机。
        penalty (float): 单次遮挡的惩罚系数。
        loss_type (str): "hinge" 使用原始归一化 hinge penalty；"l1" 使用 masked L1 penalty。
        reduction (str): "mean" | "sum" | "none"。决定返回标量或逐元素损失。
        padding_mode (str): 传递给 grid_sample 的 padding 模式。
        znear (float): 近裁剪面，避免投影时数值问题。

    Returns:
        torch.Tensor: 若 reduction != "none"，返回标量；否则返回遮挡惩罚张量，
                      形状为 (n, n, H, W)。
    """
    # 兼容 (n, H, W, 1)
    if depths.dim() == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    if depths.dim() != 3:
        raise ValueError("depths 需为形状 (n, H, W) 或 (n, H, W, 1)")

    n, H, W = depths.shape

    # 1) 反投影为 3D 点 (n, H, W, 3) -> (n, H*W, 3)
    points = depths_to_points_parallel(depthmap=depths, cameras=cameras).view(n, -1, 3)

    # 2) 将所有 3D 点复制到每个视角，计算真实深度与投影采样深度
    points_for_all_views = points.view(1, -1, 3).repeat(n, 1, 1)  # (n, n*H*W, 3)
    true_depths = cameras.p3d_cameras.get_world_to_view_transform().transform_points(
        points_for_all_views
    )[..., 2]  # (n, n*H*W)

    sampled_depths, fov_mask = get_points_depth_in_depthmap_parallel(
        pts=points_for_all_views,  # (n, n*H*W, 3)
        depthmap=depths,           # (n, H, W)
        cameras=cameras,
        padding_mode=padding_mode,
        znear=znear,
    )  # (n, n*H*W), (n, n*H*W)

    # 3) 遮挡判定：点的真实深度更近（值更小）且在视场内
    occlusion_mask = (true_depths < sampled_depths + 1e-2) & fov_mask
    if loss_type == "hinge":
        occlusion_penalty = penalty * (sampled_depths - true_depths) / (sampled_depths + 1e-5)
    elif loss_type == "l1":
        occlusion_penalty = penalty * (sampled_depths - true_depths).abs()
    else:
        raise ValueError(f"Unknown depth-order occlusion loss_type: {loss_type}")

    penalty_map = occlusion_penalty * occlusion_mask.float()  # (n, n*H*W)
    penalty_map = penalty_map.view(n, n, H, W)      # (n_source, n_target, H, W)

    if reduction == "none":
        return penalty_map
    if reduction == "sum":
        return penalty_map.sum()
    # 默认 mean
    return penalty_map.mean()
