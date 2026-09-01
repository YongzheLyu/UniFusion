import torch
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非交互式backend，适用于headless服务器
import matplotlib.pyplot as plt
import argparse
import os
import cv2


def load_image(img_path):
    """加载图像并转换为tensor"""
    img = Image.open(img_path).convert('RGB')
    return img


def preprocess(img1_batch, img2_batch):
    """预处理图像对"""
    transforms_list = transforms.Compose([
        transforms.ToTensor(),
    ])

    img1 = transforms_list(img1_batch)
    img2 = transforms_list(img2_batch)

    return img1, img2


class FlowEstimator:
    """封装RAFT模型加载和推理，支持批量前向光流计算"""

    def __init__(self, model_type='large', device=None):
        if model_type == 'large':
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights, progress=True)
        else:
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=True)

        self.model = self.model.eval()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = self.model.to(self.device)

    def _compute_single_pair(self, img1_tensor, img2_tensor):
        """计算单对图像的光流，输入为已预处理的tensor (C, H, W)"""
        img1 = img1_tensor.unsqueeze(0).to(self.device)
        img2 = img2_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            flow_predictions = self.model(img1, img2)
            flow = flow_predictions[-1]
        return flow[0].permute(1, 2, 0).cpu().numpy()

    def compute_pairs(self, img_pairs, batch_size=1):
        """
        计算多对图像的前向光流

        Args:
            img_pairs: 图像对列表 [(img1_pil, img2_pil), ...]，元素为PIL.Image
            batch_size: 若所有图像尺寸相同，支持batch forward；否则fallback到逐对

        Returns:
            flows: numpy数组列表，每个shape为 (H, W, 2)
        """
        if len(img_pairs) == 0:
            return []

        # 检查是否所有图像尺寸一致
        sizes = [img1.size for img1, _ in img_pairs]
        all_same_size = all(s == sizes[0] for s in sizes)

        flows = []

        if all_same_size and batch_size > 1:
            # Batch forward: 将img_pairs按batch_size分组
            for start in range(0, len(img_pairs), batch_size):
                end = min(start + batch_size, len(img_pairs))
                batch = img_pairs[start:end]

                img1_batch = []
                img2_batch = []
                for img1_pil, img2_pil in batch:
                    img1, img2 = preprocess(img1_pil, img2_pil)
                    img1_batch.append(img1)
                    img2_batch.append(img2)

                img1_tensor = torch.stack(img1_batch, dim=0).to(self.device)
                img2_tensor = torch.stack(img2_batch, dim=0).to(self.device)

                with torch.no_grad():
                    flow_predictions = self.model(img1_tensor, img2_tensor)
                    batch_flows = flow_predictions[-1]  # (B, 2, H, W)

                for i in range(batch_flows.size(0)):
                    flow = batch_flows[i].permute(1, 2, 0).cpu().numpy()
                    flows.append(flow)
        else:
            # Fallback: 逐对推理
            for img1_pil, img2_pil in img_pairs:
                img1, img2 = preprocess(img1_pil, img2_pil)
                flow = self._compute_single_pair(img1, img2)
                flows.append(flow)

        return flows


def compute_optical_flow(img1_path, img2_path, model_type='large'):
    """
    使用RAFT计算两帧图像之间的光流（保持向后兼容）

    Args:
        img1_path: 第一帧图像路径
        img2_path: 第二帧图像路径
        model_type: 'large' 或 'small'

    Returns:
        flow: 光流结果 (H, W, 2)
    """
    estimator = FlowEstimator(model_type=model_type)
    img1_pil = load_image(img1_path)
    img2_pil = load_image(img2_path)
    flows = estimator.compute_pairs([(img1_pil, img2_pil)], batch_size=1)
    return flows[0]


def visualize_flow(flow, save_path=None):
    """可视化光流（需要OpenCV）"""
    # 计算光流的幅度和角度
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 使用HSV色彩空间可视化
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # 角度转换为色调
    hsv[..., 1] = 255  # 饱和度
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 幅度转换为亮度

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title('Optical Flow Visualization')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    return rgb


def simple_visualize_flow(flow, save_path=None):
    """简单的光流可视化（不需要OpenCV）"""
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(flow[..., 0], cmap='RdBu')
    plt.title('Flow X')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(flow[..., 1], cmap='RdBu')
    plt.title('Flow Y')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(flow_magnitude, cmap='jet')
    plt.title('Flow Magnitude')
    plt.colorbar()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def batch_compute_flow(img_pairs, model_type='large'):
    """
    批量处理多对图像（保持向后兼容）

    Args:
        img_pairs: 图像对路径列表 [(img1_path, img2_path), ...]
        model_type: 'large' 或 'small'

    Returns:
        flows: 光流结果列表
    """
    estimator = FlowEstimator(model_type=model_type)
    pil_pairs = [(load_image(p1), load_image(p2)) for p1, p2 in img_pairs]
    return estimator.compute_pairs(pil_pairs, batch_size=1)


def warp_points(points, flow):
    """
    利用双线性插值对 flow 采样，将图像坐标前向传播。

    Args:
        points: (N, 2) float32，图像坐标 (x, y)
        flow: (H, W, 2) float32

    Returns:
        (N, 2) float32，新的图像坐标
    """
    H, W = flow.shape[:2]
    x = points[:, 0]
    y = points[:, 1]

    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    wa = ((x1 - x) * (y1 - y))[:, None]
    wb = ((x1 - x) * (y - y0))[:, None]
    wc = ((x - x0) * (y1 - y))[:, None]
    wd = ((x - x0) * (y - y0))[:, None]

    flow_interp = (
        wa * flow[y0, x0] +
        wb * flow[y1, x0] +
        wc * flow[y0, x1] +
        wd * flow[y1, x1]
    )

    return points + flow_interp


def render_tracking_frame(img, trajectories, trail_length=10, valid=None):
    """
    在背景图像上绘制 tracking 拖尾和当前点。

    Args:
        img: (H, W, 3) numpy array，BGR 背景图
        trajectories: list of list of (x, y)，每条轨迹的历史坐标
        trail_length: 拖尾长度（最近多少帧）
        valid: (N,) bool 数组，表示哪些点有效

    Returns:
        (H, W, 3) 绘制后的帧
    """
    if valid is None:
        valid = np.ones(len(trajectories), dtype=bool)

    canvas = img.copy()
    H, W = canvas.shape[:2]

    # 根据轨迹起点的 x 坐标生成颜色映射
    colors = []
    for i in range(len(trajectories)):
        if len(trajectories[i]) > 0:
            x0 = trajectories[i][0][0]
            hue = int(180 * x0 / W) % 180
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            colors.append(tuple(int(c) for c in bgr))
        else:
            colors.append((0, 255, 0))

    for i, (traj, is_valid) in enumerate(zip(trajectories, valid)):
        if not is_valid or len(traj) == 0:
            continue

        recent = np.array(traj[-trail_length:] if trail_length > 0 else traj[-1:], dtype=np.int32)

        if len(recent) > 1:
            pts = recent.reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, colors[i], 2)

        cv2.circle(canvas, tuple(recent[-1]), 3, colors[i], -1)

    return canvas


def generate_tracking_video(image_dir, flow_path, output_path, grid_step=20, trail_length=10, fps=15, verbose=False):
    """
    从已有光流和图像序列生成 2D tracking 视频。

    Args:
        image_dir: 图像目录路径
        flow_path: npz 光流文件路径
        output_path: 输出 mp4 路径
        grid_step: 采样网格间距
        trail_length: 拖尾长度
        fps: 输出视频帧率
        verbose: 是否打印详细信息
    """
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    img_files = sorted([f for f in os.listdir(image_dir)
                        if f.lower().endswith(valid_exts)])
    img_paths = [os.path.join(image_dir, f) for f in img_files]

    if len(img_paths) == 0:
        print(f"警告: {image_dir} 中没有找到图像")
        return

    data = np.load(flow_path)
    flows = data['flows']  # (N, H, W, 2)

    first_img = cv2.imread(img_paths[0])
    if first_img is None:
        print(f"警告: 无法读取图像 {img_paths[0]}")
        return
    H, W = first_img.shape[:2]

    # 若光流分辨率与图像不一致，先 resize 并缩放向量
    if flows.shape[1:3] != (H, W):
        resized_flows = []
        scale_x = W / flows.shape[2]
        scale_y = H / flows.shape[1]
        for i in range(len(flows)):
            resized = cv2.resize(flows[i], (W, H), interpolation=cv2.INTER_LINEAR)
            resized[..., 0] *= scale_x
            resized[..., 1] *= scale_y
            resized_flows.append(resized)
        flows = np.stack(resized_flows, axis=0)

    # 第 0 帧初始化网格点
    xs = np.arange(grid_step // 2, W, grid_step, dtype=np.float32)
    ys = np.arange(grid_step // 2, H, grid_step, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    trajectories = [[] for _ in range(len(points))]
    for i, p in enumerate(points):
        trajectories[i].append(p.copy().astype(np.int32).tolist())

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    valid = np.ones(len(points), dtype=bool)
    frame = render_tracking_frame(first_img, trajectories, trail_length, valid)
    writer.write(frame)

    n_flows = len(flows)
    for t in range(n_flows):
        points = warp_points(points, flows[t])

        # 边界检测
        valid = (points[:, 0] >= 0) & (points[:, 0] < W) & (points[:, 1] >= 0) & (points[:, 1] < H)

        for i in range(len(points)):
            if valid[i]:
                trajectories[i].append(points[i].copy().astype(np.int32).tolist())

        img_idx = min(t + 1, len(img_paths) - 1)
        img = cv2.imread(img_paths[img_idx])
        if img is None:
            img = first_img.copy()

        frame = render_tracking_frame(img, trajectories, trail_length, valid)
        writer.write(frame)

    writer.release()

    if verbose:
        print(f"  Tracking video shape: {H}x{W}, frames: {n_flows + 1}")
    print(f"  Tracking video saved to: {output_path}")


def process_camera(camera_dir, output_path, estimator, stride=1, batch_size=1, verbose=False,
                   vis_tracking=False, grid_step=20, trail_length=10, fps=15):
    """
    处理单个camera目录，计算相邻帧（或按stride采样）的前向光流，保存为npz

    Args:
        camera_dir: camera图像目录路径
        output_path: 输出npz路径
        estimator: FlowEstimator实例
        stride: 相邻帧对间隔
        batch_size: 批量推理大小
        verbose: 是否打印详细信息
    """
    # 读取并排序图像
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    img_files = sorted([f for f in os.listdir(camera_dir)
                        if f.lower().endswith(valid_exts)])
    img_paths = [os.path.join(camera_dir, f) for f in img_files]

    if len(img_paths) < stride + 1:
        if verbose:
            print(f"跳过 {camera_dir}: 帧数不足 (找到 {len(img_paths)} 帧，需要至少 {stride + 1} 帧)")
        return

    if verbose:
        print(f"处理 {camera_dir}: {len(img_paths)} 帧，stride={stride}，将生成 {len(img_paths) - stride} 对光流")

    # 构建图像对列表 (t, t+stride)
    img_pairs = []
    for i in range(len(img_paths) - stride):
        img1 = load_image(img_paths[i])
        img2 = load_image(img_paths[i + stride])
        img_pairs.append((img1, img2))

    # 批量计算光流
    flows = estimator.compute_pairs(img_pairs, batch_size=batch_size)

    # 堆叠为 (N, H, W, 2)
    flows_array = np.stack(flows, axis=0)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, flows=flows_array)

    if verbose:
        print(f"  光流数组 shape: {flows_array.shape}")
        print(f"  保存到: {output_path}")

    if vis_tracking:
        tracking_output = output_path.replace('_flows.npz', '_tracking.mp4')
        generate_tracking_video(
            image_dir=camera_dir,
            flow_path=output_path,
            output_path=tracking_output,
            grid_step=grid_step,
            trail_length=trail_length,
            fps=fps,
            verbose=verbose
        )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='使用RAFT模型计算光流',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Panoptic dataset processing
    parser.add_argument('--input-dir', type=str, default=None,
                        help='输入目录，包含多个camera子目录（如 panoptic_dataset/.../renamed_images/）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录，将生成 cam*_flows.npz')
    parser.add_argument('--stride', type=int, default=1,
                        help='相邻帧对间隔（如 stride=5 则计算 t->t+5 的光流）')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='同一camera内批量推理大小（要求帧尺寸相同）')

    # 基本参数（单对图像模式，向后兼容）
    parser.add_argument('--img1', type=str, default='frame1.jpg',
                        help='第一帧图像路径')
    parser.add_argument('--img2', type=str, default='frame2.jpg',
                        help='第二帧图像路径')
    parser.add_argument('--model', type=str, default='large', choices=['large', 'small'],
                        help='RAFT模型类型: large (更准确但慢) 或 small (更快但精度略低)')

    # 输出参数
    parser.add_argument('--output', type=str, default='optical_flow.npy',
                        help='输出光流文件路径 (.npy格式)，可指定完整路径')

    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化光流结果')
    parser.add_argument('--vis-output', type=str, default='flow_visualization.png',
                        help='可视化结果保存路径，可指定完整路径')
    parser.add_argument('--use-opencv-vis', action='store_true',
                        help='使用OpenCV进行HSV可视化（需要安装OpenCV）')

    # Tracking 视频参数
    parser.add_argument('--vis-tracking', action='store_true',
                        help='Panoptic模式下，处理完每个camera后自动生成2D tracking视频')
    parser.add_argument('--make-tracking-video', action='store_true',
                        help='纯离线模式：仅从已有的npz光流和图像生成tracking视频')
    parser.add_argument('--flows', type=str, default=None,
                        help='离线模式下输入的npz光流文件路径')
    parser.add_argument('--output-video', type=str, default=None,
                        help='离线模式下输出的tracking视频路径')
    parser.add_argument('--grid-step', type=int, default=20,
                        help='Tracking网格采样间距（像素）')
    parser.add_argument('--trail-length', type=int, default=10,
                        help='轨迹拖尾长度（最近多少帧），0表示只画当前点')
    parser.add_argument('--fps', type=int, default=15,
                        help='输出tracking视频的帧率')

    # 批量处理参数（向后兼容）
    parser.add_argument('--batch', type=str, default=None,
                        help='批量处理模式：指定包含图像对路径的文本文件，每行格式: img1_path img2_path')
    parser.add_argument('--batch-output-dir', type=str, default='batch_flows',
                        help='批量处理时光流输出目录')

    # 调试参数
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出模式')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='指定设备 (默认自动检测)')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 离线 tracking 视频生成模式（不计算光流）
    if args.make_tracking_video:
        if not args.input_dir or not args.flows or not args.output_video:
            raise ValueError("--make-tracking-video 需要同时指定 --input-dir、--flows 和 --output-video")
        generate_tracking_video(
            image_dir=args.input_dir,
            flow_path=args.flows,
            output_path=args.output_video,
            grid_step=args.grid_step,
            trail_length=args.trail_length,
            fps=args.fps,
            verbose=args.verbose
        )
        return

    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.verbose:
        print(f"使用设备: {device}")
        print(f"模型类型: {args.model}")

    # Panoptic dataset processing mode
    if args.input_dir and args.output_dir:
        if args.verbose:
            print(f"Panoptic模式: 输入={args.input_dir}, 输出={args.output_dir}")

        os.makedirs(args.output_dir, exist_ok=True)
        estimator = FlowEstimator(model_type=args.model, device=device)

        # 自动发现camera子目录
        camera_dirs = sorted([os.path.join(args.input_dir, d)
                              for d in os.listdir(args.input_dir)
                              if os.path.isdir(os.path.join(args.input_dir, d))])

        if args.verbose:
            print(f"发现 {len(camera_dirs)} 个camera目录: {[os.path.basename(d) for d in camera_dirs]}")

        for camera_dir in camera_dirs:
            cam_name = os.path.basename(camera_dir)
            output_path = os.path.join(args.output_dir, f"{cam_name}_flows.npz")
            process_camera(
                camera_dir=camera_dir,
                output_path=output_path,
                estimator=estimator,
                stride=args.stride,
                batch_size=args.batch_size,
                verbose=args.verbose,
                vis_tracking=args.vis_tracking,
                grid_step=args.grid_step,
                trail_length=args.trail_length,
                fps=args.fps
            )

        print(f"Panoptic处理完成，结果保存到: {args.output_dir}")
        return

    # 批量处理模式（向后兼容）
    if args.batch:
        if args.verbose:
            print(f"批量处理模式，读取图像对列表: {args.batch}")

        # 读取图像对列表
        img_pairs = []
        with open(args.batch, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        img_pairs.append((parts[0], parts[1]))

        if args.verbose:
            print(f"找到 {len(img_pairs)} 对图像")

        # 创建输出目录
        os.makedirs(args.batch_output_dir, exist_ok=True)

        # 批量计算光流
        flows = batch_compute_flow(img_pairs, model_type=args.model)

        # 保存结果
        for i, (flow, (img1, img2)) in enumerate(zip(flows, img_pairs)):
            flow_output = os.path.join(args.batch_output_dir, f'flow_{i:04d}.npy')
            np.save(flow_output, flow)

            if args.verbose:
                print(f"[{i+1}/{len(flows)}] {img1} -> {img2}")
                print(f"  光流shape: {flow.shape}")
                print(f"  保存到: {flow_output}")

        print(f"批量处理完成，结果保存到: {args.batch_output_dir}")
        return

    # 单对图像处理模式
    if args.verbose:
        print(f"处理图像对: {args.img1} -> {args.img2}")

    # 检查输入文件是否存在
    if not os.path.exists(args.img1):
        raise FileNotFoundError(f"找不到图像文件: {args.img1}")
    if not os.path.exists(args.img2):
        raise FileNotFoundError(f"找不到图像文件: {args.img2}")

    # 计算光流
    flow = compute_optical_flow(args.img1, args.img2, model_type=args.model)

    # 打印统计信息
    print(f"光流计算完成!")
    print(f"  Shape: {flow.shape}")
    print(f"  X方向范围: [{flow[..., 0].min():.2f}, {flow[..., 0].max():.2f}]")
    print(f"  Y方向范围: [{flow[..., 1].min():.2f}, {flow[..., 1].max():.2f}]")
    flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    print(f"  幅度范围: [{flow_magnitude.min():.2f}, {flow_magnitude.max():.2f}]")
    print(f"  平均幅度: {flow_magnitude.mean():.2f}")

    # 创建输出目录（如果需要）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存光流
    np.save(args.output, flow)
    print(f"光流已保存到: {args.output}")

    # 可视化
    if args.visualize:
        if args.verbose:
            print("生成可视化...")

        # 创建可视化输出目录（如果需要）
        vis_dir = os.path.dirname(args.vis_output)
        if vis_dir and not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        if args.use_opencv_vis:
            try:
                visualize_flow(flow, save_path=args.vis_output)
                print(f"可视化已保存到: {args.vis_output}")
            except ImportError:
                print("警告: 未安装OpenCV，使用简单可视化方法")
                simple_visualize_flow(flow, save_path=args.vis_output)
                print(f"可视化已保存到: {args.vis_output}")
        else:
            simple_visualize_flow(flow, save_path=args.vis_output)
            print(f"可视化已保存到: {args.vis_output}")

        plt.close('all')


if __name__ == '__main__':
    main()
