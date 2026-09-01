import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pickle
from pathlib import Path

from matcha.dm_trainers.charts_alignment_temporal import align_charts_temporal

from rich.console import Console


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal charts alignment using preprocessed data')

    # Data arguments
    parser.add_argument('-p', '--preprocessed_data', type=str, required=True,
                       help='Path to preprocessed data file (.pkl)')
    parser.add_argument('-o', '--output_path', type=str, default=None,
                       help='Output directory for aligned charts')

    # Temporal parameters
    parser.add_argument('--temporal_encoding_type', type=str, default='learned', choices=['learned', 'positional'],
                       help='Type of temporal encoding (default: learned)')
    parser.add_argument('--temporal_encoding_dim', type=int, default=8,
                       help='Dimension of temporal features (default: 8)')
    parser.add_argument('--rank', type=int, default=None,
                       help='ResField rank. Overrides alignment.resfield_rank in config when set.')
    parser.add_argument('--use_occlusion_loss', action='store_true',
                       help='Enable depth-order occlusion loss during temporal alignment.')
    parser.add_argument('--occlusion_loss_weight', type=float, default=None,
                       help='Depth-order occlusion loss weight. Overrides alignment.occlusion_loss_weight when set.')
    parser.add_argument('--depth_order_loss_type', type=str, default=None, choices=['hinge', 'l1'],
                       help='Depth-order occlusion loss type. Overrides alignment.depth_order_loss_type when set.')
    parser.add_argument('--use_ssi_loss', action='store_true',
                       help='Enable scale-and-shift invariant depth regularization.')
    parser.add_argument('--ssi_loss_weight', type=float, default=None,
                       help='SSI loss weight. Overrides alignment.ssi_loss_weight when set.')

    # Other parameters
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Starting frame index (default: 0)')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='Ending frame index (default: None, use all frames)')

    args = parser.parse_args()

    # Set console
    CONSOLE = Console(width=120)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load preprocessed data
    CONSOLE.print(f"[INFO] Loading preprocessed data from: {args.preprocessed_data}")
    with open(args.preprocessed_data, 'rb') as f:
        data = pickle.load(f)

    # Extract data
    temporal_scene_pms = data['temporal_scene_pms']
    temporal_reference_data = data['temporal_reference_data']
    temporal_masks = data['temporal_mast3r_masks']
    temporal_frame_indices = data['temporal_frame_indices']
    scale_factor = data['scale_factor']
    pm_config = data['pm_config']
    scene_config = data['scene_config']
    #masking_config = data['masking_config']
    preprocessing_args = data['args']
    print("preprocessing_args:", preprocessing_args)
    # Filter data based on start_frame and end_frame
    n_timestamps_original = len(temporal_scene_pms)
    CONSOLE.print(f"[INFO] Loaded data for {n_timestamps_original} timestamps")
    CONSOLE.print(f"[INFO] Original frame indices: {temporal_frame_indices}")

    # Find indices that match the frame range
    valid_indices = []
    for i, frame_idx in enumerate(temporal_frame_indices):
        if args.start_frame <= frame_idx:
            if args.end_frame is None or frame_idx <= args.end_frame:
                valid_indices.append(i)

    if len(valid_indices) == 0:
        raise ValueError(f"No frames found in range [{args.start_frame}, {args.end_frame if args.end_frame else 'end'}]")

    # Slice data based on valid indices
    temporal_scene_pms = [temporal_scene_pms[i] for i in valid_indices]
    temporal_reference_data = [temporal_reference_data[i] for i in valid_indices]
    # if temporal_masks is not None:
    #     temporal_masks = [temporal_masks[i] if temporal_masks[i] is not None else None for i in valid_indices]
    temporal_frame_indices = [temporal_frame_indices[i] for i in valid_indices]

    n_timestamps = len(temporal_scene_pms)
    CONSOLE.print(f"[INFO] Filtered to {n_timestamps} timestamps (from {n_timestamps_original})")
    CONSOLE.print(f"[INFO] Frame range: [{args.start_frame}, {args.end_frame if args.end_frame else 'end'}]")
    CONSOLE.print(f"[INFO] Filtered frame indices: {temporal_frame_indices}")

    print("temporal_scene_pms device:", temporal_scene_pms[0].device)
    print("temporal_reference_data device:", temporal_reference_data[0].device)
    # Ensure all pointmaps are on CPU to prevent unexpected GPU memory allocation
    for pm in temporal_scene_pms:
        pm.move_everything_to_device('cpu')

    CONSOLE.print(f"[INFO] Scale factor: {scale_factor}")

    # Set output path
    if args.output_path is None:
        preprocessed_path = Path(args.preprocessed_data)
        args.output_path = preprocessed_path.parent / "temporal_charts"
    else:
        args.output_path = Path(args.output_path)

    args.output_path.mkdir(parents=True, exist_ok=True)
    CONSOLE.print(f"[INFO] Output will be saved to: {args.output_path}")

    # === Check input data for NaN/Inf ===
    CONSOLE.print("\n[CHECK] Checking input data for NaN/Inf...")
    def check_tensor(tensor, name):
        """Check tensor for NaN/Inf and print info."""
        if isinstance(tensor, torch.Tensor):
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            has_neg_inf = (tensor == float('-inf')).any().item()
            has_pos_inf = (tensor == float('inf')).any().item()
            if has_nan or has_inf:
                print(f"\n[WARNING] {name}:")
                print(f"  - Has NaN: {has_nan}")
                print(f"  - Has Inf: {has_inf}")
                print(f"  - Has -Inf: {has_neg_inf}")
                print(f"  - Has +Inf: {has_pos_inf}")
                print(f"  - Shape: {tensor.shape}")
                print(f"  - Min: {tensor.min().item()}, Max: {tensor.max().item()}, Mean: {tensor.mean().item()}")
                # Find locations of NaN/Inf
                nan_mask = torch.isnan(tensor)
                inf_mask = torch.isinf(tensor)
                num_nan = nan_mask.sum().item()
                num_inf = inf_mask.sum().item()
                print(f"  - Num NaN: {num_nan} ({100*num_nan/tensor.numel():.4f}%)")
                print(f"  - Num Inf: {num_inf} ({100*num_inf/tensor.numel():.4f}%)")
                return True
            else:
                pass
                #print(f"[OK] {name}: No NaN/Inf found. Shape: {tensor.shape}, Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}")
        return False

    nan_inf_found = False

    # Check temporal_reference_data (深度数据)
    for t, ref_data in enumerate(temporal_reference_data):
        if isinstance(ref_data, torch.Tensor):
            if check_tensor(ref_data, f"temporal_reference_data[{t}]"):
                nan_inf_found = True
        elif isinstance(ref_data, list):
            for i, item in enumerate(ref_data):
                if isinstance(item, torch.Tensor):
                    if check_tensor(item, f"temporal_reference_data[{t}][{i}]"):
                        nan_inf_found = True

    # Check temporal_scene_pms (pointmaps)
    for t, pm in enumerate(temporal_scene_pms):
        if hasattr(pm, 'points3d'):
            if check_tensor(pm.points3d, f"pointmap[{t}].points3d"):
                nan_inf_found = True
        if hasattr(pm, 'images') and pm.images is not None:
            if check_tensor(pm.images, f"pointmap[{t}].images"):
                nan_inf_found = True

    # Check temporal_masks
    if temporal_masks is not None:
        for t, mask in enumerate(temporal_masks):
            if mask is not None:
                if check_tensor(mask, f"temporal_masks[{t}]"):
                    nan_inf_found = True

    # Check scale_factor
    if isinstance(scale_factor, (int, float)):
        if scale_factor == 0 or not (abs(scale_factor) > 1e-10 and abs(scale_factor) < 1e10):
            print(f"[WARNING] scale_factor: {scale_factor} (可能有问题)")
            nan_inf_found = True
        else:
            print(f"[OK] scale_factor: {scale_factor}")

    if nan_inf_found:
        print("\n" + "="*60)
        print("[ERROR] 发现输入数据中存在 NaN/Inf，训练可能失败！")
        print("="*60)
    else:
        print("[CHECK] 输入数据检查完成，未发现 NaN/Inf。")

    # === Align the charts temporally ===
    CONSOLE.print("\n[INFO] Starting temporal charts alignment...")

    

    # Load alignment config (we'll use the same config as preprocessing)
    import yaml
    config_path = os.path.join('configs/charts_alignment', preprocessing_args['config'] + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    align_config = config['alignment']
    if args.rank is not None:
        align_config['resfield_rank'] = args.rank
    if args.use_occlusion_loss:
        align_config['use_occlusion_loss'] = True
    if args.occlusion_loss_weight is not None:
        align_config['occlusion_loss_weight'] = args.occlusion_loss_weight
    if args.depth_order_loss_type is not None:
        align_config['depth_order_loss_type'] = args.depth_order_loss_type
    if args.use_ssi_loss:
        align_config['use_ssi_loss'] = True
    if args.ssi_loss_weight is not None:
        align_config['ssi_loss_weight'] = args.ssi_loss_weight
    #test = input("pause")
    output = align_charts_temporal(
        # Scene
        temporal_scene_pms=temporal_scene_pms,
        # Data parameters
        temporal_reference_data=temporal_reference_data,
        temporal_masks=None,
        rendering_size=pm_config['max_img_size'],
        target_scale=scene_config['target_scale'],
        # Temporal parameters
        temporal_encoding_type=args.temporal_encoding_type,
        temporal_encoding_dim=args.temporal_encoding_dim,
        # Other parameters
        verbose=True,
        return_training_losses=True,
        reprojection_matches_file=None,
        save_charts_data=True,
        charts_data_path=str(args.output_path),
        start_frame=args.start_frame,
        **align_config,
    )

    temporal_outputs, training_losses = output

    CONSOLE.print("\n===== Temporal Alignment Complete! =====")
    CONSOLE.print(f"\nProcessed frames:")
    for i, frame_idx in enumerate(temporal_frame_indices):
        CONSOLE.print(f"  - Frame {frame_idx:05d} -> Timestamp {i}")



    CONSOLE.print(f"\n[INFO] All results saved to: {args.output_path}")
