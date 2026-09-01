
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "2d-gaussian-splatting"))
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# Add the MAtCha directory to Python path
#sys.path.append(os.path.join(os.path.dirname(__file__), 'MAtCha', '2d-gaussian-splatting'))

from arguments import ModelParams
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.system_utils import searchForMaxIteration

def load_config_from_checkpoint(checkpoint_path):
    """Load configuration from checkpoint directory"""
    cfg_path = os.path.join(checkpoint_path, "cfg_args")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found at {cfg_path}")
    
    with open(cfg_path, 'r') as f:
        cfg_content = f.read()
    
    # Parse the configuration safely
    try:
        # Remove problematic Python object references
        import re
        cleaned_content = re.sub(r',\s*[_a-zA-Z]\w*=<[^>]+object\s+at\s+0x[0-9a-fA-F]+>', '', cfg_content)
        cleaned_content = re.sub(r'[_a-zA-Z]\w*=<[^>]+object\s+at\s+0x[0-9a-fA-F]+>,?\s*', '', cleaned_content)
        
        # Create a safe namespace for evaluation
        namespace = {
            'Namespace': argparse.Namespace,
            'os': os,
            'cuda': 'cuda'
        }
        
        args = eval(cleaned_content, namespace)
        return args
    except Exception as e:
        print(f"Warning: Failed to parse config file: {e}")
        # Return default configuration
        return argparse.Namespace(
            sh_degree=3,
            source_path='',
            model_path=checkpoint_path,
            images='images',
            resolution=-1,
            white_background=False,
            data_device='cuda',
            eval=False,
            render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'],
            kplanes_config={
                'grid_dimensions': 2,
                'input_coordinate_dim': 4,
                'output_coordinate_dim': 16,
                'resolution': [64, 64, 64, 150]
            },
            multires=[1, 2],
            defor_depth=0,
            net_width=128,
            plane_tv_weight=0.0002,
            time_smoothness_weight=0.001,
            l1_time_planes=0.0001,
            no_dx=False,
            no_grid=False,
            no_do=False,
            no_dshs=False,
            no_ds=False,
            no_dr=False,
            empty_voxel=False,
            render_process=False,
            static_mlp=False,
            bounds=1.6,
            timebase_pe=4,
            grid_pe=0,
            posebase_pe=10,
            scale_rotation_pe=2,
            opacity_pe=2,
            timenet_width=64,
            timenet_output=32,
            apply_rotation=False
        )

def export_static_pointcloud(checkpoint_path, output_path, frame_number, time_normalization=150.0):
    """
    Export static 3DGS point cloud for a specific time frame
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        output_path: Path to save the output PLY file
        frame_number: Frame number (0-150 for typical 150-frame sequences)
        time_normalization: Normalization factor for time (default 150.0 for 150-frame sequences)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Exporting frame {frame_number} to: {output_path}")
    
    # Load configuration
    args = load_config_from_checkpoint(checkpoint_path)
    args.model_path = checkpoint_path
    
    # Initialize Gaussian model
    gaussians = GaussianModel(args.sh_degree, args)
    
    # Find the latest iteration if not specified
    iteration = searchForMaxIteration(os.path.join(checkpoint_path, "point_cloud"))
    print(f"Using iteration: {iteration}")
    
    # Load the trained model
    print("Loading point cloud...")
    point_cloud_path = os.path.join(checkpoint_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(point_cloud_path)
    
    print("Loading deformation model...")
    deformation_path = os.path.join(checkpoint_path, "deformation", f"iteration_{iteration}")
    gaussians.load_model(deformation_path)
    
    # Calculate normalized time value
    time_value = frame_number / time_normalization
    print(f"Time value: {time_value} (frame {frame_number} / {time_normalization})")
    
    # Create time tensor
    time_tensor = torch.tensor([[time_value]], dtype=torch.float32, device='cuda').repeat(gaussians.get_xyz.shape[0],1)
    print(time_tensor.shape)
    print("Computing deformation for specified time frame...")
    with torch.no_grad():
        # Get deformed positions
        
        #deformed_xyz = gaussians.get_deformed_xyz(time_tensor)
        
        # Get other deformed properties
        deformation_result = gaussians._deformation(
            gaussians.get_xyz, 
            gaussians.get_scaling, 
            gaussians.get_rotation, 
            gaussians.get_opacity,
            gaussians.get_features, 
            time_tensor
        )
        
        if isinstance(deformation_result, tuple) and len(deformation_result) == 5:
            deformed_xyz, deformed_scaling, deformed_rotation, deformed_opacity, deformed_shs = deformation_result
        else:
            # Fallback: use original properties if deformation doesn't return all values
            deformed_xyz = gaussians.get_deformed_xyz(time_tensor)
            deformed_scaling = gaussians._scaling
            deformed_rotation = gaussians._rotation  
            deformed_opacity = gaussians._opacity
            deformed_shs = None
    
    print(f"Exporting static point cloud with {deformed_xyz.shape[0]} points...")
    
    # Create a temporary GaussianModel to save the deformed point cloud
    temp_gaussians = GaussianModel(args.sh_degree, args)
    
    # Set the deformed properties
    temp_gaussians._xyz = deformed_xyz
    temp_gaussians._scaling = deformed_scaling if deformed_scaling is not None else gaussians._scaling
    temp_gaussians._rotation = deformed_rotation if deformed_rotation is not None else gaussians._rotation
    temp_gaussians._opacity = deformed_opacity if deformed_opacity is not None else gaussians._opacity
    temp_gaussians._features_dc = gaussians._features_dc
    temp_gaussians._features_rest = gaussians._features_rest
    #print()
    # Save the deformed point cloud
    temp_gaussians.save_ply(os.path.join(output_path,f"static_ply/iter{iteration}_f{frame_number}.ply"))
    
    print(f"Successfully exported static point cloud to: {output_path}")
    print(f"Points: {deformed_xyz.shape[0]}")
    print(f"Frame: {frame_number} (time: {time_value:.3f})")

def main():
    parser = argparse.ArgumentParser(description="Export static 3DGS point cloud for specific time frame")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Path to checkpoint directory (e.g., dataset_for_training_11_05/free_gaussians)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output PLY file path")
    parser.add_argument("--frame", type=int, required=True,
                       help="Frame number (0-150 for typical sequences)")
    parser.add_argument("--time_normalization", type=float, default=10.0,
                       help="Time normalization factor (default: 150.0 for 150-frame sequences)")
    parser.add_argument("--iteration", type=int, default=None,
                       help="Specific iteration to load (default: latest)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path does not exist: {args.checkpoint_path}")
        return 1
    
    if args.frame < 0:
        print(f"Error: Frame number must be non-negative: {args.frame}")
        return 1
    
    try:
        export_static_pointcloud(
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            frame_number=args.frame,
            time_normalization=args.time_normalization
        )
        return 0
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
