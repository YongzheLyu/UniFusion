from typing import List
import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from torch.nn.functional import normalize as torch_normalize
from torch.utils.tensorboard import SummaryWriter

from matcha.dm_scene.parallel_aligner import (
    ParallelAligner,
    ChartsEncodingParams,
    DepthEncodingParams,
    MLPParams,
)
from matcha.dm_deformation.multi_mlp_temporal_resfield import DeformationMultiMLPResField
from matcha.dm_deformation.multi_mlp import initialize_multi_mlp_weights
from matcha.dm_regularization.depth_order_occlusion import depth_order_occlusion_loss
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

class TemporalEncodingParams():
    def __init__(
        self,
        encoding_dim:int=32,
        encoding_type:str='learned',  # 'learned' or 'positional'
        max_freq:int=5,  # Only used for positional encoding
        initialization_range:float=1e-2,
    ):
        self.encoding_dim = encoding_dim
        self.encoding_type = encoding_type
        self.max_freq = max_freq
        self.initialization_range = initialization_range


class ParallelAlignerTemporal(ParallelAligner):
    """Temporal version of ParallelAligner for multi-timestamp optimization.

    This class extends ParallelAligner to handle multiple timestamps by:
    1. Treating T timestamps with N charts each as T×N total charts
    2. Using n_heads=N in the MLP (shared across timestamps for the same view)
    3. Injecting temporal information through time features in the MLP

    Key differences from ParallelAligner:
    - depths: (T×N, H, W) instead of (N, H, W)
    - MLP has n_heads=N but processes all T×N charts by grouping by view
    - Time features are concatenated to spatial features before MLP
    """
    def __init__(
        self,
        depths:torch.Tensor,  # (T×N, H, W)
        cameras,  # CamerasWrapper with T×N cameras
        n_timestamps:int,
        n_charts_per_timestamp:int,
        timestamp_indices:torch.Tensor=None,  # (T×N,) - timestamp index for each chart
        temporal_encoding_params:TemporalEncodingParams=TemporalEncodingParams(),
        charts_encoding_params:ChartsEncodingParams=ChartsEncodingParams(),
        depth_encoding_params:DepthEncodingParams=DepthEncodingParams(),
        mlp_params:MLPParams=MLPParams(),
        use_learnable_depth_encoding:bool=True,
        learnable_depth_encoding_mode:str='add',
        device='cuda',
        **kwargs
    ) -> None:
        """Initialize temporal parallel aligner.

        Args:
            depths (torch.Tensor): Shape (T×N, H, W) where T=n_timestamps, N=n_charts_per_timestamp
            cameras: CamerasWrapper containing T×N cameras
            n_timestamps (int): Number of timestamps T
            n_charts_per_timestamp (int): Number of charts per timestamp N
            timestamp_indices (torch.Tensor, optional): Shape (T×N,) indicating which timestamp each chart belongs to.
                If None, assumes charts are ordered as [t0_c0, t0_c1, ..., t0_cN-1, t1_c0, ..., tT-1_cN-1]
            temporal_encoding_params (TemporalEncodingParams, optional): Parameters for temporal encoding.
            charts_encoding_params (ChartsEncodingParams, optional): Parameters for charts encoding.
            depth_encoding_params (DepthEncodingParams, optional): Parameters for depth encoding.
            mlp_params (MLPParams, optional): Parameters for MLP.
            use_learnable_depth_encoding (bool, optional): Whether to use learnable depth encoding. Defaults to True.
            learnable_depth_encoding_mode (str, optional): Mode for depth encoding ('add', 'concatenate', etc). Defaults to 'add'.
            device (str, optional): Device for computation. Defaults to 'cuda'.
        """
        self.n_timestamps = n_timestamps
        self.n_charts_per_timestamp = n_charts_per_timestamp
        self.resfield_rank = kwargs.pop('resfield_rank', 8)
        # temporal_encoding_params no longer needed for ResFields

        # Create timestamp indices if not provided
        if timestamp_indices is None:
            # Assume charts are ordered: [t0_c0, ..., t0_cN, t1_c0, ..., t1_cN, ...]
            timestamp_indices = torch.repeat_interleave(
                torch.arange(n_timestamps),
                n_charts_per_timestamp
            )

        # Initialize parent class
        # Note: We pass depths with shape (T×N, H, W) but will override the MLP creation
        super(ParallelAlignerTemporal, self).__init__(
            depths=depths,
            cameras=cameras,
            charts_encoding_params=charts_encoding_params,
            depth_encoding_params=depth_encoding_params,
            mlp_params=mlp_params,
            use_learnable_depth_encoding=use_learnable_depth_encoding,
            learnable_depth_encoding_mode=learnable_depth_encoding_mode,
            device=device,
            n_timestamps=n_timestamps,
            **kwargs
        )
        #self.device = "cuda"
        # Register as buffer (will be moved to device automatically)
        # Must be done AFTER super().__init__() call
        self.register_buffer('timestamp_indices', timestamp_indices.to(self.device))
        self.pm_h = kwargs.get('pm_h', None)
        self.pm_w = kwargs.get('pm_w', None)

        # ==== Camera parameters (cam2world) ====
        # Extract per-chart camera pose (cam2world) parameters and store as a 9D vector:
        # 6D rotation representation (first two columns of R) + 3D translation T.
        # These raw parameters will be fed directly into the deformation MLP as part of additional_input.
        # Extract cam2world (view-to-world) matrices for all cameras
        try:
            # world_to_view: (N, 4, 4)
            w2v_mats = self.cameras.p3d_cameras.get_world_to_view_transform().get_matrix().to(self.device)
            v2w_mats = torch.inverse(w2v_mats)  # (N, 4, 4)
            # Use 6D continuous rotation representation: take first two columns of R (Zhou et al. 2019)
            R_full = v2w_mats[:, :3, :3]                 # (N, 3, 3)
            R_6d = R_full[:, :, :2].reshape(-1, 6)       # (N, 6)
            T = v2w_mats[:, :3, 3]                       # (N, 3)
            cam_params = torch.cat([R_6d, T], dim=-1)    # (N, 9)
        except Exception as e:
            # Fallback: if get_matrix is not available, use identity poses
            n_cams = self.cameras.p3d_cameras.R.shape[0]
            cam_params = torch.zeros(n_cams, 9, device=self.device)
            print(f"[WARNING] Failed to extract cam2world for camera encoding: {e}. Using zeros.")

        # Register raw camera parameters as buffer (not trainable)
        self.register_buffer('camera_params', cam_params)

        # Override: Create temporal MLP with ResFields
        self._create_temporal_mlp(mlp_params, charts_encoding_params, depth_encoding_params)
        print("Using ResFields for temporal modeling - no temporal encoding needed")
        print("finish init")
        #a = input("pause")
    def _create_temporal_mlp(self, mlp_params, charts_encoding_params, depth_encoding_params):
        """Create MLP with temporal feature support.

        Override the MLP created by parent class to use DeformationMultiMLPResField.
        """
        
        scene_radius = mlp_params.scene_radius_factor * self.cameras.get_spatial_extent()
        deformation_radius = mlp_params.deformation_radius_factor * self.cameras.get_spatial_extent()

        # additional_input can contain:
        #   (1) depth encodings (when learnable_depth_encoding_mode == 'concatenate')
        #   (2) camera parameters (9D: 6D rotation + 3D translation)
        #   (3) ray directions (3D: normalized ray vectors)
        
        # Calculate additional_input_dim
        _additional_input_dim = 0
        if self.use_learnable_depth_encoding and self.learnable_depth_encoding_mode == 'concatenate':
            _additional_input_dim += charts_encoding_params.encoding_dim
        # Add ray directions (3D)
        _additional_input_dim += 0
        
        # Remove the parent class's deformation module completely
        # This ensures proper cleanup before creating the new temporal MLP
        if hasattr(self, 'deformation'):
            # Remove from module registry first (this is the key step)
            if 'deformation' in self._modules:
                old_module = self._modules.pop('deformation')
                del old_module  # Ensure old module is released
            # Remove attribute reference (safe after removing from _modules)
            if hasattr(self, 'deformation'):
                delattr(self, 'deformation')
        # Use temporal MLP with ResFields for n_heads = n_charts_per_timestamp
        # This means the same view across different timestamps share MLP parameters
        self.deformation = DeformationMultiMLPResField(
            n_heads=self.n_charts_per_timestamp,  # Key: n_heads = N, not T×N
            n_layer=mlp_params.n_deformation_layers,
            layer_size=mlp_params.deformation_layer_size,
            input_dim=charts_encoding_params.encoding_dim,
            output_dim=1 if self.predict_deformations_along_rays else 3,
            additional_input_dim=_additional_input_dim,
            data_input_range_min=-scene_radius,
            data_input_range_max=scene_radius,
            mlp_input_range_min=-1.,
            mlp_input_range_max=1.,
            output_range_min=-deformation_radius,
            output_range_max=deformation_radius,
            non_linearity=torch.nn.ReLU(),
            final_non_linearity=None if mlp_params.no_final_linearity else torch.nn.Sigmoid(),
            positional_encoding=None,
            output_points=False,
            # ResFields parameters
            resfield_layers=None,  # Apply ResFields to all layers
            resfield_rank=self.resfield_rank,
            resfield_capacity=self.n_timestamps,  # One capacity slot per timestamp
            resfield_mode='lookup',
            resfield_compression='vm',
            resfield_fuse_mode='add',
            resfield_coeff_ratio=1.0,
        ).cuda()

        # Initialize MLP weights
        initialize_multi_mlp_weights(self.deformation, std=None)

    def compute_verts_for_charts(self, chart_indices, profile=False):
        """Compute deformed vertices for specified charts only (memory efficient).

        Args:
            chart_indices: List or tensor of chart indices to process
            profile: If True, print detailed timing information for each step

        Returns:
            verts: (len(chart_indices), H*W, 3) - deformed vertices
        """
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            func_start_time = time.time()
            step_times = {}

        device = "cuda"
        #device = "cpu"
        
        # Step 0: Input preparation - keep on GPU to avoid transfers
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.time()
        
        # Convert to tensor and keep on GPU if possible
        if isinstance(chart_indices, list):
            chart_indices = torch.tensor(chart_indices, device="cpu")
        elif chart_indices.device != device:
            chart_indices = chart_indices.to("cpu")

        n_charts = len(chart_indices)
        _charts_encoding_dim = self.charts_encoding_params.encoding_dim

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times['0_input_prep'] = time.time() - step_start

        # 1. Get spatial encodings for specified charts (直接在GPU上计算)
        # 需要按时间戳分组处理，因为charts_encoding一次只能处理一个时间戳的charts
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.time()
        
        # Pre-compute timestamp masks and indices to avoid repeated computation
        timestamps_in_batch = torch.unique(chart_indices // self.n_charts_per_timestamp, sorted=True)
        timestamp_masks = {}  # Cache masks for each timestamp
        timestamp_indices_dict = {}  # Cache indices for each timestamp
        for t in timestamps_in_batch:
            t_val = t.item()
            mask = (chart_indices // self.n_charts_per_timestamp) == t_val
            timestamp_masks[t_val] = mask
            timestamp_indices_dict[t_val] = chart_indices[mask]

        if len(timestamps_in_batch) == 1:
            # 只有一个时间戳，直接处理
            t_val = timestamps_in_batch[0].item()
            timestamp_chart_indices = timestamp_indices_dict[t_val]
            pts_uv_subset = self._pts_uv[timestamp_chart_indices].to(device)  # (n_charts, H*W, 2)
            if profile:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                encoding_start = time.time()
            encodings = self.charts_encoding(pts_uv_subset).reshape(n_charts, -1, _charts_encoding_dim)
            if profile:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_times['1a_charts_encoding_single'] = time.time() - encoding_start
        else:
            # 多个时间戳，需要分组处理
            encodings = torch.zeros(n_charts, self.pm_h * self.pm_w, _charts_encoding_dim, device=device)
            encoding_total_time = 0.0
            for t in timestamps_in_batch:
                t_val = t.item()
                timestamp_chart_mask = timestamp_masks[t_val]
                timestamp_chart_indices = timestamp_indices_dict[t_val]

                if len(timestamp_chart_indices) == 0:
                    continue

                # Extract data for this timestamp and get encodings
                pts_uv_t = self._pts_uv[timestamp_chart_indices.cpu()].to(device)  # (n_views_in_t, H*W, 2)
                if profile:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    encoding_start = time.time()
                encodings_t_ = self.charts_encoding(pts_uv_t).reshape(len(timestamp_chart_indices), -1, _charts_encoding_dim)
                if profile:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    encoding_total_time += time.time() - encoding_start

                # Put back into the correct positions in encodings tensor
                encodings[timestamp_chart_mask] = encodings_t_
            if profile:
                step_times['1a_charts_encoding_multi'] = encoding_total_time
                step_times['1a_charts_encoding_avg_per_timestamp'] = encoding_total_time / len(timestamps_in_batch)

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.time()

        # Weight encodings with confidence if needed
        if self.weight_encodings_with_confidence:
            # Fix: Remove detach() to maintain gradient flow (critical for convergence)
            conf_weights = (self.confidence[chart_indices].to(device) - 1.).view(n_charts, -1, 1)
            conf_weights = 1. - torch.exp(-conf_weights**2 / 2)
            encodings = encodings * conf_weights

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times['1b_confidence_weighting'] = time.time() - step_start if self.weight_encodings_with_confidence else 0.0

        # 2. Add depth encoding if needed
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.time()
        
        depth_encodings = None  # Initialize for concatenate mode
        if self.use_learnable_depth_encoding:
            if len(timestamps_in_batch) == 1:
                # 只有一个时间戳，直接处理
                t_val = timestamps_in_batch[0].item()
                timestamp_chart_indices = timestamp_indices_dict[t_val]
                depth_coords_subset = self.depth_coords[timestamp_chart_indices].to(device)
                if profile:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    depth_encoding_start = time.time()
                depth_encodings = self.depth_encoding(depth_coords_subset).reshape(n_charts, -1, _charts_encoding_dim)
                if profile:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    step_times['2a_depth_encoding_single'] = time.time() - depth_encoding_start
            else:
                # 多个时间戳，需要分组处理 - reuse pre-computed masks
                depth_encodings = torch.zeros(n_charts, self.pm_h * self.pm_w, _charts_encoding_dim, device=device)
                depth_encoding_total_time = 0.0
                for t in timestamps_in_batch:
                    t_val = t.item()
                    timestamp_chart_mask = timestamp_masks[t_val]
                    timestamp_chart_indices = timestamp_indices_dict[t_val]

                    if len(timestamp_chart_indices) == 0:
                        continue

                    # Extract depth coords for this timestamp and get encodings
                    depth_coords_t = self.depth_coords[timestamp_chart_indices.cpu()].to(device)
                    if profile:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        depth_encoding_start = time.time()
                    depth_encodings_t = self.depth_encoding(depth_coords_t).reshape(len(timestamp_chart_indices), -1, _charts_encoding_dim)
                    if profile:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        depth_encoding_total_time += time.time() - depth_encoding_start
                    # Put back into the correct positions in depth_encodings tensor
                    depth_encodings[timestamp_chart_mask] = depth_encodings_t
                if profile:
                    step_times['2a_depth_encoding_multi'] = depth_encoding_total_time
                    step_times['2a_depth_encoding_avg_per_timestamp'] = depth_encoding_total_time / len(timestamps_in_batch)

            if profile:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                merge_start = time.time()

            if not (self.use_meta_mlp or self.use_lora_mlp):
                if self.learnable_depth_encoding_mode == 'add':
                    encodings = encodings + depth_encodings
                elif self.learnable_depth_encoding_mode == 'multiply':
                    encodings = encodings * depth_encodings
                elif self.learnable_depth_encoding_mode == 'replace':
                    encodings = depth_encodings
                elif self.learnable_depth_encoding_mode == 'concatenate':
                    # Keep depth_encodings separate for concatenate mode
                    pass  # depth_encodings will be used as additional_input later
                elif self.learnable_depth_encoding_mode == 'adaln':
                    encodings = encodings + 0.001 * self.adaln(encodings, depth_encodings)

            if profile:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_times['2b_depth_encoding_merge'] = time.time() - merge_start

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times['2_depth_encoding_total'] = time.time() - step_start if self.use_learnable_depth_encoding else 0.0

        # 3. Get timestamp indices for ResFields - pre-compute all at once
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.time()

        timestamp_indices_subset = self.timestamp_indices[chart_indices].to(device)  # Ensure on device

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times['3_timestamp_indices'] = time.time() - step_start

        # 4. Group charts by timestamp and process each timestamp
        # IMPORTANT: Reuse encodings already computed above to avoid duplicate computation
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.time()
        n_verts_per_chart = self.pm_h * self.pm_w
        deformations = torch.zeros(n_charts, n_verts_per_chart, self.deformation.output_dim, device=device)
        
        # Get rays_subset for all charts (will be used as additional_input)
        rays_subset = self._rays[chart_indices].to(device)  # (n_charts, H*W, 3)
        
        deformation_total_time = 0.0
        # Process each timestamp in the batch - reuse pre-computed masks and indices
        for t in timestamps_in_batch:
            t_val = t.item()
            timestamp_chart_mask = timestamp_masks[t_val]
            timestamp_chart_indices = timestamp_indices_dict[t_val]

            if len(timestamp_chart_indices) == 0:
                continue

            # ✅ REUSE already computed encodings instead of recalculating
            # Extract the pre-computed encodings for this timestamp
            encodings_t = encodings[timestamp_chart_mask]  # (n_views_in_t, H*W, encoding_dim)
            
            # Extract rays for this timestamp
            rays_subset_t = rays_subset[timestamp_chart_mask]  # (n_views_in_t, H*W, 3)
            
            # Prepare additional_input: depth encodings (if concatenate mode) + rays
            additional_input_t = None
            
            #additional_input_t = rays_subset_t  # (n_views_in_t, H*W, 3)
            
            # Get timestamp indices for ResFields - pre-computed
            timestamp_indices_t = self.timestamp_indices[timestamp_chart_indices].to(device)  # (n_views_in_t,)

            # Forward through MLP using ResFields with timestamp indices
            if profile:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                deformation_start = time.time()

            deformations_t = self.deformation(
                encodings_t,
                frame_id=timestamp_indices_t,
                additional_input=additional_input_t
            )  # (n_views_in_t, H*W, output_dim)
            if profile:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                deformation_total_time += time.time() - deformation_start

            # Store results in the correct positions
            deformations[timestamp_chart_mask] = deformations_t

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times['4_deformation_total'] = time.time() - step_start
            step_times['4a_deformation_mlp_only'] = deformation_total_time
            step_times['4a_deformation_mlp_avg_per_timestamp'] = deformation_total_time / len(timestamps_in_batch) if len(timestamps_in_batch) > 0 else 0.0
      
        
        # 7. Handle ray-aligned deformations
        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.time()
        
        rays_subset = self._rays[chart_indices].to(device)
        predict_deformations_along_rays = deformations.shape[-1] == 1
        no_residual = False
    
        if no_residual:
            deformed_verts = deformations * torch_normalize(rays_subset, dim=-1)
            deformed_verts = deformed_verts.reshape(n_charts, -1, 3)
        else:
            # 如果deformations最后一维是1维，沿着射线方向处理
            if predict_deformations_along_rays:
                if self.predict_in_disparity_space:
                    mlp_output_scale = (self.deformation.output_range_max - self.deformation.output_range_min) / 2
                    deformations = deformations / mlp_output_scale * rays_subset
                else:
                    deformations = deformations * torch_normalize(rays_subset, dim=-1)
            else:
                if self.predict_in_disparity_space:
                    mlp_output_scale = (self.deformation.output_range_max - self.deformation.output_range_min) / 2
                    deformations = deformations / mlp_output_scale * rays_subset.norm(dim=-1, keepdim=True)
            
            verts_subset = self._verts[chart_indices].to(device)  # (n_charts, H*W, 3)
            deformed_verts = verts_subset.reshape(n_charts, -1, 3) + deformations.reshape(n_charts, -1, 3)

        

        return deformed_verts

    @property
    def verts_deformations(self):
        """Compute vertex deformations with temporal features.

        Override parent method to inject time features into MLP.
        按timestamp分批处理encoding以避免显存问题。
        """
        device = self.device
        _charts_encoding_dim = self.charts_encoding_params.encoding_dim

        # 按照timestamp分批处理，避免一次性处理所有charts导致的显存问题
        all_deformations = []

        for t in range(self.n_timestamps):
            # 1. Get charts for this timestamp
            start_idx = t * self.n_charts_per_timestamp
            end_idx = (t + 1) * self.n_charts_per_timestamp
            chart_indices = torch.arange(start_idx, end_idx, device=device)

            # 2. Get spatial encodings for this timestamp (直接在GPU上计算)
            pts_uv_t = self._pts_uv[chart_indices].cuda() # (N, H, W, 2)
            encodings = self.charts_encoding(pts_uv_t).reshape(self.n_charts_per_timestamp, -1, _charts_encoding_dim)

            # Weight encodings with confidence if needed
            if self.weight_encodings_with_confidence:
                conf_weights = (self.confidence[chart_indices].detach() - 1.).view(self.n_charts_per_timestamp, -1, 1)
                conf_weights = 1. - torch.exp(-conf_weights**2 / 2)
                encodings = encodings * conf_weights.cuda()

            # 3. Add depth encoding if needed
            additional_input = None
            if self.use_learnable_depth_encoding:
                depth_coords_t = self.depth_coords[chart_indices.cpu()].cuda()
                depth_encodings = self.depth_encoding(depth_coords_t).reshape(self.n_charts_per_timestamp, -1, _charts_encoding_dim)

                if not (self.use_meta_mlp or self.use_lora_mlp):
                    if self.learnable_depth_encoding_mode == 'add':
                        encodings = encodings + depth_encodings
                    elif self.learnable_depth_encoding_mode == 'multiply':
                        encodings = encodings * depth_encodings
                    elif self.learnable_depth_encoding_mode == 'replace':
                        encodings = depth_encodings
                    elif self.learnable_depth_encoding_mode == 'concatenate':
                        additional_input = depth_encodings  # (N, H*W, enc_dim)
                    elif self.learnable_depth_encoding_mode == 'adaln':
                        encodings = encodings + 0.001 * self.adaln(encodings, depth_encodings)
                    else:
                        raise ValueError(f"learnable_depth_encoding_mode must be either 'add', 'replace', \
                                            or 'concatenate', and not {self.learnable_depth_encoding_mode}.")

            # 3b. Get rays for this timestamp
            rays_t = self._rays[chart_indices.cpu()].to(device)  # (N, H*W, 3)
            
            # Combine additional_input: depth encodings (if concatenate mode) + rays
            if additional_input is not None:
                # additional_input already contains depth_encodings in concatenate mode
                additional_input = torch.cat([additional_input, rays_t], dim=-1)  # (N, H*W, enc_dim + 3)
            else:
                additional_input = rays_t  # (N, H*W, 3)

            # 4. Get timestamp indices for ResFields
            timestamp_indices_t = self.timestamp_indices[chart_indices]  # (N,)

            # 5. Forward through temporal MLP with ResFields
            deformations_t = self.deformation(
                encodings,
                frame_id=timestamp_indices_t.cuda(),
                #additional_input=additional_input
            )  # (N, H*W, output_dim)

            # Save deformation maps as heatmaps for this timestamp
            deformation_save_dir = "deformation_heatmaps"
            os.makedirs(deformation_save_dir, exist_ok=True)

            # deformations_t shape: (8, 147456, 1) -> reshape to (8, 288, 512)
            n_charts, n_verts, output_dim = deformations_t.shape
            height = self.pm_h
            width = self.pm_w

            # Ensure the total number of vertices matches
            assert n_verts == height * width, f"Expected {height * width} vertices, got {n_verts}"

            # Reshape and save each chart's deformation map
            for chart_idx in range(n_charts):
                deformation_map = deformations_t[chart_idx, :, 0].reshape(height, width).cpu().detach().numpy()

                # Create heatmap
                # plt.figure(figsize=(10, 6))
                # plt.imshow(deformation_map, cmap='jet', aspect='auto')
                # plt.colorbar()
                # plt.title(f'Timestamp {t}, Chart {chart_idx} - Deformation Map')
                # plt.axis('off')

                # # Save the heatmap
                # save_path = os.path.join(deformation_save_dir, f'timestamp_{t:03d}_chart_{chart_idx:02d}_deformation.png')
                # plt.savefig(save_path, bbox_inches='tight', dpi=150)
                # plt.close()

                #print(f"Saved deformation heatmap: {save_path}")

            all_deformations.append(deformations_t)

        # 6. Concatenate all timestamps
        deformations = torch.cat(all_deformations, dim=0).to(device)  # (T×N, H*W, output_dim)

        # 7. Handle ray-aligned deformations
        predict_deformations_along_rays = deformations.shape[-1] == 1
        rays = self._rays.to(device)
        no_residual = False
        if no_residual:
            deformations = deformations * torch_normalize(rays, dim=-1)
            #deformed_verts = deformed_verts.reshape(n_charts, -1, 3)
            #print(deformed_verts.shape)
        else:
            if predict_deformations_along_rays or self.predict_in_disparity_space:
                if rays is None:
                    raise ValueError("Rays must be provided if we are predicting deformations along rays or in disparity space.")
            if predict_deformations_along_rays:
                if self.predict_in_disparity_space:
                    mlp_output_scale = (self.deformation.output_range_max - self.deformation.output_range_min) / 2
                    deformations = deformations / mlp_output_scale * rays
                else:
                    deformations = deformations * torch_normalize(rays, dim=-1)
            else:
                if self.predict_in_disparity_space:
                    mlp_output_scale = (self.deformation.output_range_max - self.deformation.output_range_min) / 2
                    deformations = deformations / mlp_output_scale * rays.norm(dim=-1, keepdim=True)

        return deformations

    @property
    def verts(self):
        """Override parent verts to handle temporal deformations correctly."""
        #print("vertsdsadasdsadadsadsadsadsadasddsadasd")
        deformations = self.verts_deformations.view(self.n_pm, self.pm_h, self.pm_w, 3)  # (T×N, H*W, 3)
        print(deformations.mean(),self._verts.mean())
        return self._verts + deformations

    def loss(self, reference_depths, pred_depths, masks=None, chart_indices=None):
        """Override parent's loss to support chart_indices for memory-efficient mode.

        When chart_indices is provided, only use confidence for the specified charts
        instead of all charts.

        Args:
            reference_depths: Reference depth values
            pred_depths: Predicted depth values
            masks: Optional masks
            chart_indices: Optional tensor of chart indices (for memory-efficient mode)

        Returns:
            loss: Computed loss value
        """
        if self.using_pts_as_reference:
            # For point cloud reference, use parent's implementation
            return super().loss(reference_depths, pred_depths, masks)

        diff = pred_depths - reference_depths
        diff = diff.abs()

        if self.use_learnable_confidence:
            # Get confidence only for specified charts if chart_indices is provided
            if chart_indices is not None:
                # chart_indices may be on GPU, need CPU for indexing
                if chart_indices.device.type == 'cuda':
                    chart_indices_cpu = chart_indices.cpu()
                else:
                    chart_indices_cpu = chart_indices
                confidence = self.confidence[chart_indices_cpu].to(pred_depths.device)
            else:
                confidence = self.confidence.to(pred_depths.device)

            diff = confidence * diff - self.confidence_weighting * torch.log(confidence)
            #diff = diff

        if masks is not None:
            diff = masks * diff

        return diff.mean()

    def reset_encodings(self):
        """Reset encodings for temporal version with ResFields.

        Override parent method - no temporal encoding needed for ResFields.
        """
        # Call parent's reset_encodings to handle charts_encoding and depth_encoding
        super().reset_encodings()

        # No temporal encoding reset needed for ResFields

    @torch.no_grad()
    def reset_mlp(self, std=None):
        """Reset MLP for temporal version.

        Override parent method to use DeformationMultiMLPResField instead of DeformationMultiMLP.
        """
        if self.use_meta_mlp or self.use_lora_mlp:
            # For meta/lora MLP, use parent's implementation
            super().reset_mlp(std=std)
        else:
            # Use temporal MLP
            scene_radius = self.mlp_params.scene_radius_factor * self.cameras.get_spatial_extent()
            deformation_radius = self.mlp_params.deformation_radius_factor * self.cameras.get_spatial_extent()
            
            # additional_input: depth encodings (concatenate mode) + ray directions (3D)
            _additional_input_dim = 0
            if self.use_learnable_depth_encoding and self.learnable_depth_encoding_mode == 'concatenate':
                _additional_input_dim += self.charts_encoding_params.encoding_dim
            # Add ray directions (3D)
            _additional_input_dim += 0
            
            # Save old module attributes before removing it
            old_n_layer = self.mlp_params.n_deformation_layers
            old_layer_size = self.mlp_params.deformation_layer_size
            old_input_dim = self.charts_encoding_params.encoding_dim
            old_output_dim = 1 if self.predict_deformations_along_rays else 3
            old_non_linearity = torch.nn.ReLU()
            old_final_non_linearity = None if self.mlp_params.no_final_linearity else torch.nn.Sigmoid()
            old_positional_encoding = None
            
            if hasattr(self, 'deformation') and self.deformation is not None:
                old_n_layer = self.deformation.n_layer
                old_layer_size = self.deformation.layer_size
                old_input_dim = self.deformation.input_dim
                old_output_dim = self.deformation.output_dim
                old_non_linearity = self.deformation.non_linearity
                old_final_non_linearity = self.deformation.final_non_linearity
                old_positional_encoding = self.deformation._positional_encoding
            
            # Remove old module if exists
            if hasattr(self, 'deformation'):
                if 'deformation' in self._modules:
                    old_module = self._modules.pop('deformation')
                    del old_module
                if hasattr(self, 'deformation'):
                    delattr(self, 'deformation')
            
            # Create new temporal MLP with ResFields
            self.deformation = DeformationMultiMLPResField(
                n_heads=self.n_charts_per_timestamp,  # Key: n_heads = N, not T×N
                n_layer=old_n_layer,
                layer_size=old_layer_size,
                input_dim=old_input_dim,
                output_dim=old_output_dim,
                additional_input_dim=_additional_input_dim,
                data_input_range_min=-scene_radius,
                data_input_range_max=scene_radius,
                mlp_input_range_min=-1.,
                mlp_input_range_max=1.,
                output_range_min=-deformation_radius,
                output_range_max=deformation_radius,
                non_linearity=old_non_linearity,
                final_non_linearity=old_final_non_linearity,
                positional_encoding=old_positional_encoding,
                output_points=False,
                # ResFields parameters
                resfield_layers=None,  # Apply ResFields to all layers
                resfield_rank=self.resfield_rank,
                resfield_capacity=self.n_timestamps,  # One capacity slot per timestamp
                resfield_mode='lookup',
                resfield_compression='vm',
                resfield_fuse_mode='add',
                resfield_coeff_ratio=1.0,
            ).cuda()
            
            # Initialize MLP weights
            initialize_multi_mlp_weights(self.deformation, std=std)

    def optimize(
        self,
        reference_data:torch.Tensor,
        masks:torch.Tensor=None,
        gradient_masks:torch.Tensor=None,
        n_iterations=600,
        use_gradient_loss=True,
        use_hessian_loss=False,
        use_normal_loss=True,
        use_curvature_loss=False,
        use_matching_loss=False,
        use_reprojection_loss=False,
        use_occlusion_loss=False,
        matching_thr=None,
        use_confidence_in_matching_loss=False,
        matching_update_iters=None,
        gradient_loss_weight=10.,
        hessian_loss_weight=100.,
        normal_loss_weight=2.,
        curvature_loss_weight=1.,
        matching_loss_weight=1.,
        reprojection_loss_weight=2.,
        occlusion_loss_weight=5.0,
        depth_order_loss_type="hinge",
        encodings_lr:float=1e-2,
        mlp_lr:float=1e-3,
        confidence_lr:float=1e-3,
        lr_update_iters=[300],
        lr_update_factor:float=0.1,
        use_lr_scheduler:bool=True,  # Enable learning rate scheduler
        lr_scheduler_type:str='cosine',  # 'cosine', 'step', or 'exponential'
        lr_scheduler_gamma:float=0.95,  # For step/exponential scheduler
        lr_scheduler_T_max:int=None,  # For cosine scheduler (None = n_iterations)
        verbose:bool=True,
        match_to_img:torch.Tensor=None,
        match_to_pix:torch.Tensor=None,
        reprojection_loss_power:float=0.5,
        regularize_chart_encodings_norms:bool=False,
        chart_encodings_norm_loss_weight:float=0.5,
        use_total_variation_on_depth_encodings:bool=False,
        total_variation_on_depth_encodings_weight:float=1.0,
        # SSI loss between deformed depths and initial depths
        use_ssi_loss:bool=False,
        ssi_loss_weight:float=2.0,
        # New parameters for memory efficiency
        batch_timestamps:int=2,  # Number of timestamps to process per iteration (None = all)
        enable_memory_efficient:bool=True,  # Enable memory-efficient training
        accumulation_steps:int=2,  # Number of gradient accumulation steps (1 = no accumulation)
        # TensorBoard logging
        use_tensorboard:bool=True,  # Enable TensorBoard logging
        log_dir:str=None,  # Directory for TensorBoard logs (None = auto-generate)
    ):
        """Temporal version of optimize with per-frame matching and memory-efficient batching.

        This override implements:
        1. Per-timestamp matching to avoid memory issues when matching all T×N cameras together
        2. Optional memory-efficient batching by processing subset of timestamps per iteration
        3. Learning rate scheduling for better convergence

        Args:
            batch_timestamps: If specified and enable_memory_efficient=True, randomly sample
                this many timestamps per iteration. Default is None (process all).
            enable_memory_efficient: If True, enables batching mode for memory efficiency.
            use_lr_scheduler: If True, use learning rate scheduler for automatic LR decay.
            lr_scheduler_type: Type of scheduler ('cosine', 'step', or 'exponential').
            lr_scheduler_gamma: Decay factor for step/exponential scheduler (default: 0.95).
            lr_scheduler_T_max: Maximum iterations for cosine scheduler (None = n_iterations).
            use_tensorboard: If True, log losses to TensorBoard. Defaults to True.
            log_dir: Directory for TensorBoard logs. If None, auto-generates based on timestamp.
        """
        from matcha.dm_modules.matcher_3d import Matcher3D
        from matcha.dm_utils.image import img_grad, img_hessian
        from matcha.dm_utils.rendering import depth2normal_parallel, normal2curv_parallel
        from matcha.pointmap.mast3r import get_minimal_projections_diffs
        from tqdm import tqdm
        import random

        # Setup memory-efficient mode
        if enable_memory_efficient:
            if batch_timestamps is None:
                batch_timestamps = 4  # Default to 4 timestamps per batch
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Memory Efficient Mode] Enabled")
                print(f"  > Batch size: {batch_timestamps} timestamps per iteration")
                print(f"  > Total timestamps: {self.n_timestamps}")
                print(f"  > Charts per timestamp: {self.n_charts_per_timestamp}")
                print(f"  > Memory saving: ~{100*(1 - batch_timestamps/self.n_timestamps):.1f}%")
                print(f"{'='*60}\n")
        else:
            batch_timestamps = self.n_timestamps  # Process all timestamps

        train_losses = []
        if verbose:
            print(f"Starting temporal optimization...")

        # Preprocessing of reference depths
        if reference_data[0].shape[-1] == 3:
            print("Using a list of 3D points as reference for fitting depth maps.")
            self.using_pts_as_reference = True
            try:
                if isinstance(reference_data, list):
                    reference_pts = torch.stack(reference_data, dim=0)
                reference_depths = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(reference_pts)[..., 2].flatten()
                print(f"\nReference points has shape {reference_pts.shape[0]}.")
                print(f"Reference points depths has shape {reference_depths.shape[0]}.")
                consistent_n_points = True
            except:
                print("Converting list of 3D points to tensor failed. The number of points in each chart is not consistent.")
                print("Each chart will be processed separately.")
                print("[TODO] Implement a simple padding mechanism to handle this case.")
                reference_pts = reference_data
                reference_depths = []
                for i_depth in range(len(reference_data)):
                    reference_pts_i = reference_pts[i_depth]
                    reference_pts_depth_i = self.cameras.p3d_cameras[i_depth].get_world_to_view_transform().transform_points(reference_pts_i)[..., 2]
                    reference_depths.append(reference_pts_depth_i)
                    print(f"\nDepth {i_depth} has {reference_pts_i.shape[0]} reference points.")
                reference_depths = torch.cat(reference_depths, dim=0).flatten()
                print(f"Reference points depths has shape {reference_depths.shape}.")
                consistent_n_points = False
            self.reference_pts = reference_pts
        else:
            print("Using depth maps as reference for fitting depth maps.")
            self.using_pts_as_reference = False
            reference_depths = reference_data
            if masks is not None:
                print("Using masks for optimization.")
                assert masks.shape == reference_depths.shape
            if gradient_masks is not None:
                print("Using gradient masks for optimization.")
                assert gradient_masks.shape == self._depths[..., :-1, :-1].shape
        use_matching_loss = True
        # Prepare per-timestamp matchers if needed
        if use_matching_loss:
            if self.using_pts_as_reference:
                raise NotImplementedError("Matching loss is not implemented yet for point clouds.")

            print(f"[INFO] Creating {self.n_timestamps} separate matchers for per-frame matching...")
            matchers = []
            if matching_thr is None:
                matching_thr = self.cameras.get_spatial_extent() / 20.

            # Create one matcher per timestamp
            for t in range(self.n_timestamps):
                # Get cameras and depths for this timestamp
                chart_indices = list(range(t * self.n_charts_per_timestamp, (t + 1) * self.n_charts_per_timestamp))

                # Create a camera wrapper for this timestamp
                if hasattr(self.cameras, 'p3d_cameras'):
                    timestamp_p3d_cameras = self.cameras.p3d_cameras[chart_indices]
                    # Create a temporary CamerasWrapper
                    from matcha.dm_scene.cameras import CamerasWrapper
                    timestamp_cameras = CamerasWrapper.from_p3d_cameras(timestamp_p3d_cameras.cuda(), self.pm_w, self.pm_h)
                else:
                    # Fallback: assume cameras is already a list
                    timestamp_cameras_list = [self.cameras[i] for i in chart_indices]
                    timestamp_cameras = CamerasWrapper.from_p3d_cameras(timestamp_cameras_list, self.pm_w, self.pm_h)

                timestamp_reference_depths = reference_depths[chart_indices].cuda()

                # Create matcher for this timestamp
                matcher_t = Matcher3D(cameras=timestamp_cameras, reference_depths=timestamp_reference_depths)
                matcher_t.match(matching_thr)
                matchers.append(matcher_t)
                verbose = False
                # 可视化匹配结果（可选）
                if verbose:
                    vis_dir = os.path.join("./visualize_matcher", f'matcher_visualizations_timestamp_{t}')
                    try:
                        matcher_t.visualize_all(output_dir=vis_dir, min_consensus=2)
                        print(f"[INFO] Saved matcher visualizations for timestamp {t} to: {vis_dir}")
                    except Exception as e:
                        print(f"[WARNING] Failed to generate visualizations for timestamp {t}: {e}")
                # import sys
                # sys.exit(1)

            print(f"[INFO] Created {len(matchers)} matchers successfully.")

        # Prepare for optimization
        self.prepare_for_optimization(
            encodings_lr=encodings_lr,
            mlp_lr=mlp_lr,
            confidence_lr=confidence_lr,
            lr_update_iters=lr_update_iters,
            lr_update_factor=lr_update_factor,
            verbose=verbose,
        )
        
        # Setup learning rate scheduler
        lr_scheduler = None
        if use_lr_scheduler:
            if lr_scheduler_type == 'cosine':
                from torch.optim.lr_scheduler import CosineAnnealingLR
                T_max = lr_scheduler_T_max if lr_scheduler_T_max is not None else n_iterations
                lr_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=T_max,
                    eta_min=5e-6  # Minimum learning rate
                )
                if verbose:
                    print(f"[INFO] Using CosineAnnealingLR scheduler (T_max={T_max}, eta_min=1e-6)")
            elif lr_scheduler_type == 'step':
                from torch.optim.lr_scheduler import StepLR
                # Use step size based on lr_update_iters if available
                step_size = lr_update_iters[0] if len(lr_update_iters) > 0 else n_iterations // 3
                lr_scheduler = StepLR(
                    self.optimizer,
                    step_size=step_size,
                    gamma=lr_scheduler_gamma
                )
                if verbose:
                    print(f"[INFO] Using StepLR scheduler (step_size={step_size}, gamma={lr_scheduler_gamma})")
            elif lr_scheduler_type == 'exponential':
                from torch.optim.lr_scheduler import ExponentialLR
                lr_scheduler = ExponentialLR(
                    self.optimizer,
                    gamma=lr_scheduler_gamma
                )
                if verbose:
                    print(f"[INFO] Using ExponentialLR scheduler (gamma={lr_scheduler_gamma})")
            else:
                if verbose:
                    print(f"[WARNING] Unknown scheduler type '{lr_scheduler_type}', disabling scheduler")
                use_lr_scheduler = False
        
        # 显存优化：清理初始化过程中的临时变量
        torch.cuda.empty_cache()
        
        # Setup TensorBoard logging
        writer = None
        if use_tensorboard:
            if log_dir is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = f"runs/temporal_alignment_{timestamp}"
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            if verbose:
                print(f"[INFO] TensorBoard logging enabled: {log_dir}")
                print(f"[INFO] View logs with: tensorboard --logdir {log_dir}")
        
        if verbose:
            print(f"[INFO] Device: {self.device}")
            print(f"[INFO] Memory-efficient optimizations enabled")
        if use_normal_loss or use_curvature_loss:
            if not enable_memory_efficient:
                _normals = depth2normal_parallel(self._depths.cuda(), self.cameras.to("cuda"))
                self.initial_normals = _normals
        else:
            self.initial_normals = None
        if use_curvature_loss:
            if not enable_memory_efficient:
                _curvatures = normal2curv_parallel(_normals, mask=torch.ones_like(_normals, dtype=torch.bool))
                self.initial_curvatures = _curvatures
        else:
            self.initial_curvatures = None

        # Optimization loop
        desc = "Aligning charts (temporal" + (" - memory efficient)" if enable_memory_efficient else ")")
        progress_bar = tqdm(range(n_iterations), desc=desc)
        for i_iter in range(n_iterations):
            iter_start_time = time.time()

            # Manual learning rate updates (only if not using scheduler, or as additional updates)
            if i_iter in lr_update_iters and not use_lr_scheduler:
                lr_update_start = time.time()
                if verbose:
                    print("\n[INFO] Updating learning rates (manual)...")

                # Print previous learning rates
                if verbose:
                    print("   > Previous learning rates:")
                    for param_group in self.optimizer.param_groups:
                        print(f"      > {param_group['name']}: {param_group['lr']}")

                # Update learning rates
                for param_group in self.optimizer.param_groups:
                    if (param_group['name'].endswith('encodings')
                        or param_group['name'].startswith('deformation')
                    ):
                        param_group['lr'] = param_group['lr'] * lr_update_factor

                # Print updated learning rates
                if verbose:
                    print("   > Updated learning rates:")
                    for param_group in self.optimizer.param_groups:
                        print(f"      > {param_group['name']}: {param_group['lr']}")
                lr_update_time = time.time() - lr_update_start
                if verbose:
                    print(f"[TIMING] Learning rate update: {lr_update_time:.4f}s")
            
            # Print current learning rates periodically (when using scheduler)
            if use_lr_scheduler and lr_scheduler is not None and i_iter % 50 == 0 and verbose:
                print(f"\n[INFO] Iteration {i_iter} - Current learning rates:")
                for param_group in self.optimizer.param_groups:
                    print(f"   > {param_group['name']}: {param_group['lr']:.6e}")

            # [Memory Efficient] Randomly sample timestamps if enabled
            data_prep_start = time.time()
            if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                timestamp_sampling_start = time.time()
                # Randomly sample batch_timestamps from all timestamps
                sampled_timestamps = random.sample(range(self.n_timestamps), batch_timestamps)
                sampled_timestamps = sorted(sampled_timestamps)  # Sort for easier debugging

                # Get chart indices for sampled timestamps (直接在GPU上创建)
                chart_indices = []
                for t in sampled_timestamps:
                    start_idx = t * self.n_charts_per_timestamp
                    end_idx = (t + 1) * self.n_charts_per_timestamp
                    chart_indices.extend(range(start_idx, end_idx))
                chart_indices = torch.tensor(chart_indices, device=self.device, dtype=torch.long)
                timestamp_sampling_time = time.time() - timestamp_sampling_start

                verts_computation_start = time.time()
                #pause = input("pause")
                # Compute deformed vertices only for sampled charts
                _deformed_verts_batch = self.compute_verts_for_charts(chart_indices)
                verts_computation_time = time.time() - verts_computation_start

                depths_computation_start = time.time()
                # Compute deformed depths for sampled charts
                _deformed_depths_list = []
                
                for idx, t in enumerate(sampled_timestamps):
                    start_batch_idx = idx * self.n_charts_per_timestamp
                    end_batch_idx = (idx + 1) * self.n_charts_per_timestamp
                    verts_t = _deformed_verts_batch[start_batch_idx:end_batch_idx].reshape(
                        self.n_charts_per_timestamp, -1, 3
                    )

                    # Get cameras for this timestamp
                    cam_start = t * self.n_charts_per_timestamp
                    cam_end = (t + 1) * self.n_charts_per_timestamp

                    depths_t = self.cameras.p3d_cameras[list(range(cam_start, cam_end))].get_world_to_view_transform().cuda().transform_points(verts_t)[..., 2]
                    _deformed_depths_list.append(depths_t.reshape(self.n_charts_per_timestamp, self.pm_h, self.pm_w))

                _deformed_depths_batch = torch.cat(_deformed_depths_list, dim=0)
                depths_computation_time = time.time() - depths_computation_start

                data_transfer_start = time.time()
                # Get reference data for sampled charts (确保在GPU上)
                reference_depths_batch = reference_depths[chart_indices.cpu()].to(self.device) if reference_depths.device != self.device else reference_depths[chart_indices]
                masks_batch = masks[chart_indices.cpu()].to(self.device) if masks is not None and masks.device != self.device else (masks[chart_indices] if masks is not None else None)
                data_transfer_time = time.time() - data_transfer_start

                if verbose and i_iter % 10 == 0:
                    print(f"[TIMING] Memory-efficient data prep: {time.time() - data_prep_start:.4f}s")
                    print(f"  - Timestamp sampling: {timestamp_sampling_time:.4f}s")
                    print(f"  - Verts computation: {verts_computation_time:.4f}s")
                    print(f"  - Depths computation: {depths_computation_time:.4f}s")
                    print(f"  - Data transfer: {data_transfer_time:.4f}s")

            else:
                # Original behavior: process all timestamps (保持数据在GPU上)
                full_data_prep_start = time.time()
                chart_indices = None  # Not used
                _deformed_verts = self.verts

                depths_full_start = time.time()
                _deformed_depths_list = []
                for t in range(self.n_timestamps):
                    start_idx = t * self.n_charts_per_timestamp
                    end_idx = (t + 1) * self.n_charts_per_timestamp
                    cam_indices = list(range(start_idx, end_idx))
                    verts_t = _deformed_verts[start_idx:end_idx].reshape(self.n_charts_per_timestamp, -1, 3)
                    depths_t = self.cameras.p3d_cameras[cam_indices].get_world_to_view_transform().cuda().transform_points(verts_t)[..., 2]
                    _deformed_depths_list.append(depths_t.reshape(self.n_charts_per_timestamp, self.pm_h, self.pm_w))
                _deformed_depths_batch = torch.cat(_deformed_depths_list, dim=0)
                
                depths_full_time = time.time() - depths_full_start

                data_transfer_full_start = time.time()
                reference_depths_batch = reference_depths.to(self.device) if reference_depths.device != self.device else reference_depths
                masks_batch = masks.to(self.device) if masks is not None and masks.device != self.device else masks
                data_transfer_full_time = time.time() - data_transfer_full_start

                if verbose and i_iter % 10 == 0:
                    print(f"[TIMING] Full data prep: {time.time() - full_data_prep_start:.4f}s")
                    print(f"  - Depths computation (all timestamps): {depths_full_time:.4f}s")
                    print(f"  - Data transfer: {data_transfer_full_time:.4f}s")

            # Compute loss (数据已经在GPU上，无需额外移动)
            loss_computation_start = time.time()
            # 传入 chart_indices 以便只使用对应的 confidence
            # INSERT_YOUR_CODE
            # 只保存第一张 reference_depth 和 pred_depth 为归一化热力图图片到指定目录
           
            # save_vis_dir = "./depth_visualization"
            # os.makedirs(save_vis_dir, exist_ok=True)

            # def save_depth_heatmap_first(depth_tensor, save_path, mask_tensor=None):
            #     # depth_tensor: (N, H, W) or (H, W)
            #     depth_np = depth_tensor.detach().cpu().numpy()
            #     if depth_np.ndim == 2:
            #         img = depth_np
            #     else:
            #         img = depth_np[0]
            #     if mask_tensor is not None:
            #         mask_np = mask_tensor.detach().cpu().numpy()
            #         if mask_np.ndim == 2:
            #             mask_img = mask_np
            #         elif mask_np.ndim == 3:
            #             mask_img = mask_np[0]
            #         else:
            #             mask_img = None
            #         if mask_img is not None and mask_img.shape == img.shape:
            #             valid = img[mask_img > 0]
            #         else:
            #             valid = img.flatten()
            #     else:
            #         valid = img.flatten()
            #     if valid.size > 0:
            #         vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
            #         if vmax - vmin < 1e-6:
            #             vmin, vmax = valid.min(), valid.max() + 1e-3
            #     else:
            #         vmin, vmax = 0.0, 1.0
            #     plt.figure(figsize=(6, 4))
            #     plt.imshow(np.clip((img - vmin) / max(vmax - vmin, 1e-6), 0, 1), cmap='jet')
            #     plt.axis('off')
            #     plt.tight_layout()
            #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            #     plt.close()

            # # Save first reference depth
            # save_depth_heatmap_first(
            #     reference_depths_batch,
            #     os.path.join(save_vis_dir, "reference_depth_0.png"),
            #     mask_tensor=masks_batch
            # )
            # # Save first predicted (deformed) depth
            # save_depth_heatmap_first(
            #     _deformed_depths_batch,
            #     os.path.join(save_vis_dir, "predicted_depth_0.png"),
            #     mask_tensor=masks_batch
            # )
            # sys.exit()
            # Get initial depths for current batch (reused for SSI, gradient, hessian, normal losses)
            if chart_indices is not None:
                initial_depths_batch = self._depths[chart_indices.cpu()].to(_deformed_depths_batch.device)
            else:
                initial_depths_batch = self._depths.to(_deformed_depths_batch.device)
            
            loss = self.loss(reference_depths=reference_depths_batch.cuda(), pred_depths=_deformed_depths_batch, masks=masks_batch, chart_indices=chart_indices)
            _loss = loss.detach().item()
            loss_computation_time = time.time() - loss_computation_start

            # Scale-and-Shift Invariant (SSI) loss between deformed depths and initial depths
            ssi_loss_val = 0.0
            if use_ssi_loss and hasattr(self, "_depths"):
                
                # Simple vectorized SSI loss computation
                # Flatten to (B, H*W)
                B, H, W = _deformed_depths_batch.shape
                pred_flat = _deformed_depths_batch.view(B, -1)  # (B, H*W)
                target_flat = initial_depths_batch.view(B, -1)  # (B, H*W)
                
                # Create mask if needed
                if masks_batch is not None:
                    mask_flat = (masks_batch > 0).float().view(B, -1)  # (B, H*W)
                else:
                    mask_flat = ((target_flat > 0) & torch.isfinite(target_flat) & torch.isfinite(pred_flat)).float()
                
                # Count valid pixels per sample
                n_valid = mask_flat.sum(dim=1, keepdim=True) + 1e-8  # (B, 1)
                
                # Compute means per sample (masked)
                pred_masked = pred_flat * mask_flat
                target_masked = target_flat * mask_flat
                p_mean = pred_masked.sum(dim=1, keepdim=True) / n_valid  # (B, 1)
                t_mean = target_masked.sum(dim=1, keepdim=True) / n_valid  # (B, 1)
                
                # Center the data
                p_centered = (pred_flat - p_mean) * mask_flat  # (B, H*W)
                t_centered = (target_flat - t_mean) * mask_flat  # (B, H*W)
                
                # Compute scale and bias per sample
                p_var = (p_centered ** 2).sum(dim=1, keepdim=True) + 1e-8  # (B, 1)
                cov_pt = (p_centered * t_centered).sum(dim=1, keepdim=True)  # (B, 1)
                s = cov_pt / p_var  # (B, 1)
                bias = t_mean - s * p_mean  # (B, 1)
                
                # Align prediction and compute L1 loss
                p_aligned = s * pred_flat + bias  # (B, H*W)
                l1_diff = (p_aligned - target_flat).abs() * mask_flat  # (B, H*W)
                ssi_loss = (l1_diff.sum(dim=1) / n_valid.squeeze(1)).mean()  # scalar
                
                loss = loss + ssi_loss_weight * ssi_loss
                ssi_loss_val = float(ssi_loss.detach().item())

            if use_gradient_loss:
                gradient_loss_start = time.time()
                if gradient_masks is not None:
                    # Need to index gradient_masks too if in batch mode
                    if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                        gradient_masks_batch = gradient_masks[chart_indices]
                    else:
                        gradient_masks_batch = gradient_masks
                    grad_loss = gradient_loss_weight * (gradient_masks_batch * (
                        img_grad(_deformed_depths_batch) - img_grad(initial_depths_batch)
                    )).abs().mean()
                else:
                    grad_loss = gradient_loss_weight * (
                        img_grad(_deformed_depths_batch) - img_grad(initial_depths_batch)
                    ).abs().mean()
                loss += grad_loss
                grad_loss = grad_loss.detach().item()
                gradient_loss_time = time.time() - gradient_loss_start

            if use_hessian_loss:
                hessian_loss_start = time.time()
                if gradient_masks is not None:
                    if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                        gradient_masks_batch = gradient_masks[chart_indices]
                    else:
                        gradient_masks_batch = gradient_masks
                    hess_loss = hessian_loss_weight * (gradient_masks_batch * (
                        img_hessian(_deformed_depths_batch) - img_hessian(initial_depths_batch)
                    )).abs().mean()
                else:
                    hess_loss = hessian_loss_weight * (
                        img_hessian(_deformed_depths_batch) - img_hessian(initial_depths_batch)
                    ).abs().mean()
                loss += hess_loss
                hess_loss = hess_loss.detach().item()
                hessian_loss_time = time.time() - hessian_loss_start

            if use_normal_loss:
                normal_loss_start = time.time()
                # Only compute normals for the batch
                if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                    # Extract camera matrices directly for batch
                    chart_indices_cpu = chart_indices.cpu()
                    world_view_transforms_batch = torch.stack([self.cameras.gs_cameras[int(i)].world_view_transform for i in chart_indices_cpu])
                    full_proj_transforms_batch = torch.stack([self.cameras.gs_cameras[int(i)].full_proj_transform for i in chart_indices_cpu])
                    # Reuse initial_depths_batch (already on correct device)
                    world_view_transforms_batch = world_view_transforms_batch.to(initial_depths_batch.device)
                    full_proj_transforms_batch = full_proj_transforms_batch.to(initial_depths_batch.device)
                    _normals_batch = depth2normal_parallel(
                        initial_depths_batch,
                        world_view_transforms=world_view_transforms_batch,
                        full_proj_transforms=full_proj_transforms_batch
                    )
                    #print(_deformed_depths_batch.device)
                    _deformed_normals_batch = depth2normal_parallel(
                        _deformed_depths_batch,
                        world_view_transforms=world_view_transforms_batch,
                        full_proj_transforms=full_proj_transforms_batch
                    )

                else:
                    _normals_batch = _normals
                    _deformed_normals_batch = depth2normal_parallel(_deformed_depths_batch, self.cameras)

                normal_loss = normal_loss_weight * (1. - torch.sum(_normals_batch * _deformed_normals_batch, dim=-1)).mean()
                loss += normal_loss
                normal_loss = normal_loss.detach().item()
                normal_loss_time = time.time() - normal_loss_start

            if use_curvature_loss:
                curvature_loss_start = time.time()
                if not use_normal_loss:
                    if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                        # Extract camera matrices directly for batch
                        chart_indices_cpu = chart_indices.cpu()
                        world_view_transforms_batch = torch.stack([self.cameras.gs_cameras[int(i)].world_view_transform for i in chart_indices_cpu])
                        full_proj_transforms_batch = torch.stack([self.cameras.gs_cameras[int(i)].full_proj_transform for i in chart_indices_cpu])
                        _deformed_normals_batch = depth2normal_parallel(
                            _deformed_depths_batch,
                            world_view_transforms=world_view_transforms_batch,
                            full_proj_transforms=full_proj_transforms_batch
                        )
                    else:
                        _deformed_normals_batch = depth2normal_parallel(_deformed_depths_batch, self.cameras)

                _deformed_curvatures_batch = normal2curv_parallel(_deformed_normals_batch, mask=torch.ones_like(_deformed_normals_batch, dtype=torch.bool))

                if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                    _curvatures_batch = normal2curv_parallel(_normals_batch, mask=torch.ones_like(_normals_batch, dtype=torch.bool))
                else:
                    _curvatures_batch = _curvatures

                curv_loss = curvature_loss_weight * (_curvatures_batch - _deformed_curvatures_batch).abs().mean()
                loss += curv_loss
                curv_loss = curv_loss.detach().item()
                curvature_loss_time = time.time() - curvature_loss_start

            if use_matching_loss:
                matching_loss_start = time.time()
                # Per-timestamp matching loss
                matching_loss_total = 0.
                if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                    # For memory-efficient mode, only compute for sampled timestamps
                    # Use the same sampled_timestamps that were used for this iteration
                    for t, timestamp in enumerate(sampled_timestamps):
                        # t is the index in the batch (0, 1, 2, ...)
                        # timestamp is the original timestamp number
                        start_idx = t * self.n_charts_per_timestamp
                        end_idx = (t + 1) * self.n_charts_per_timestamp
                        chart_indices_t = list(range(start_idx, end_idx))
                        
                        # Get deformed depths for this timestamp from the batch
                        timestamp_deformed_depths = _deformed_depths_batch[chart_indices_t]

                        # Compute reprojection errors for this timestamp
                        # Use matchers[timestamp] because matchers are indexed by original timestamp number
                        reprojection_errors, fov_mask = matchers[timestamp].compute_reprojection_errors(depths=timestamp_deformed_depths)
                        reprojection_errors = reprojection_errors * fov_mask * matchers[timestamp].reference_matches  # (N, N, h, w)

                        if use_confidence_in_matching_loss:
                            # Get confidence for the original timestamp indices
                            start_timestamp_idx = timestamp * self.n_charts_per_timestamp
                            end_timestamp_idx = (timestamp + 1) * self.n_charts_per_timestamp
                            timestamp_indices_t = list(range(start_timestamp_idx, end_timestamp_idx))
                            timestamp_confidence = self.confidence[timestamp_indices_t].detach()
                            reprojection_errors = reprojection_errors * timestamp_confidence[None]  # (N, N, h, w)

                        matching_loss_total += reprojection_errors.mean()
                    
                    # Average over sampled timestamps
                    matching_loss = matching_loss_weight * matching_loss_total / len(sampled_timestamps)
                else:
                    # Process all timestamps
                    for t in range(self.n_timestamps):
                        chart_indices_t = list(range(t * self.n_charts_per_timestamp, (t + 1) * self.n_charts_per_timestamp))
                        timestamp_deformed_depths = _deformed_depths_batch[chart_indices_t]

                        # Compute reprojection errors for this timestamp
                        reprojection_errors, fov_mask = matchers[t].compute_reprojection_errors(depths=timestamp_deformed_depths)
                        reprojection_errors = reprojection_errors * fov_mask * matchers[t].reference_matches  # (N, N, h, w)

                        if use_confidence_in_matching_loss:
                            timestamp_confidence = self.confidence[chart_indices_t].detach()
                            reprojection_errors = reprojection_errors * timestamp_confidence[None]  # (N, N, h, w)

                        matching_loss_total += reprojection_errors.mean()

                    # Average over timestamps
                    matching_loss = matching_loss_weight * matching_loss_total / self.n_timestamps
                
                loss += matching_loss
                matching_loss = matching_loss.detach().item()
                matching_loss_time = time.time() - matching_loss_start

            if use_reprojection_loss and match_to_img is not None and match_to_pix is not None:
                reprojection_loss_start = time.time()
                # Use batch vertices if in memory-efficient mode
                if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                    # Only compute for sampled charts
                    minimal_projections_diffs = get_minimal_projections_diffs(
                        points3d=_deformed_verts_batch.reshape(-1, 3),
                        cameras=self.cameras,
                        match_to_img=match_to_img,
                        match_to_pix=match_to_pix,
                        loss_power=reprojection_loss_power,
                    )
                else:
                    minimal_projections_diffs = get_minimal_projections_diffs(
                        points3d=_deformed_verts,
                        cameras=self.cameras,
                        match_to_img=match_to_img,
                        match_to_pix=match_to_pix,
                        loss_power=reprojection_loss_power,
                    )
                reprojection_loss = reprojection_loss_weight * minimal_projections_diffs.mean()
                loss += reprojection_loss
                reprojection_loss = reprojection_loss.detach().item()
                reprojection_loss_time = time.time() - reprojection_loss_start

            if use_occlusion_loss:
                occlusion_loss_start = time.time()
                # Compute occlusion loss per timestamp (each timestamp has its own set of cameras)
                occlusion_loss_total = 0.
                if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                    # For memory-efficient mode, only compute for sampled timestamps
                    # Use the same sampled_timestamps that were used for this iteration
                    for t, timestamp in enumerate(sampled_timestamps):
                        start_idx = t * self.n_charts_per_timestamp
                        end_idx = (t + 1) * self.n_charts_per_timestamp
                        start_timestamp_idx = timestamp * self.n_charts_per_timestamp
                        end_timestamp_idx = (timestamp + 1) * self.n_charts_per_timestamp
                        chart_indices_t = list(range(start_idx, end_idx))
                        timestamp_indices_t = list(range(start_timestamp_idx, end_timestamp_idx))
                        # Get cameras and depths for this timestamp
                        if hasattr(self.cameras, 'p3d_cameras'):
                            
                            #print(timestamp_indices_t)
                            #print(len(self.cameras.p3d_cameras))
                            timestamp_p3d_cameras = self.cameras.p3d_cameras[timestamp_indices_t]
                            from matcha.dm_scene.cameras import CamerasWrapper
                            timestamp_cameras = CamerasWrapper.from_p3d_cameras(timestamp_p3d_cameras.cuda(), self.pm_w, self.pm_h)
                        else:
                            print("no p3d cameras")
                            timestamp_cameras_list = [self.cameras[i] for i in timestamp_indices_t]
                            from matcha.dm_scene.cameras import CamerasWrapper
                            timestamp_cameras = CamerasWrapper.from_p3d_cameras(timestamp_cameras_list, self.pm_w, self.pm_h)
                        #print(len(_deformed_depths_batch), chart_indices_t)
                        timestamp_deformed_depths = _deformed_depths_batch[chart_indices_t]
                        occlusion_loss_t = depth_order_occlusion_loss(
                            depths=timestamp_deformed_depths,
                            cameras=timestamp_cameras,
                            penalty=5.0,
                            loss_type=depth_order_loss_type,
                            reduction="mean",
                            padding_mode="zeros",
                            znear=self.znear if hasattr(self, 'znear') else 1e-6,
                        )
                        occlusion_loss_total += occlusion_loss_t
                    
                    # Average over sampled timestamps
                    occlusion_loss = occlusion_loss_weight * occlusion_loss_total / len(sampled_timestamps)
                   # print("occlusion_loss: ", occlusion_loss)
                else:
                    # Process all timestamps
                    for t in range(self.n_timestamps):
                        start_idx = t * self.n_charts_per_timestamp
                        end_idx = (t + 1) * self.n_charts_per_timestamp
                        chart_indices_t = list(range(start_idx, end_idx))
                        
                        # Get cameras and depths for this timestamp
                        if hasattr(self.cameras, 'p3d_cameras'):
                            timestamp_p3d_cameras = self.cameras.p3d_cameras[chart_indices_t]
                            from matcha.dm_scene.cameras import CamerasWrapper
                            timestamp_cameras = CamerasWrapper.from_p3d_cameras(timestamp_p3d_cameras.cuda(), self.pm_w, self.pm_h)
                        else:
                            timestamp_cameras_list = [self.cameras[i] for i in chart_indices_t]
                            from matcha.dm_scene.cameras import CamerasWrapper
                            timestamp_cameras = CamerasWrapper.from_p3d_cameras(timestamp_cameras_list, self.pm_w, self.pm_h)
                        
                        timestamp_deformed_depths = _deformed_depths_batch[chart_indices_t]
                        occlusion_loss_t = depth_order_occlusion_loss(
                            depths=timestamp_deformed_depths,
                            cameras=timestamp_cameras,
                            penalty=4.0,
                            loss_type=depth_order_loss_type,
                            reduction="mean",
                            padding_mode="zeros",
                            znear=self.znear if hasattr(self, 'znear') else 1e-6,
                        )
                        occlusion_loss_total += occlusion_loss_t
                    
                    # Average over all timestamps
                    occlusion_loss = occlusion_loss_weight * occlusion_loss_total / self.n_timestamps
                
                loss += occlusion_loss
                occlusion_loss = occlusion_loss.detach().item()
                occlusion_loss_time = time.time() - occlusion_loss_start

            if regularize_chart_encodings_norms:
                # Only regularize sampled charts in memory-efficient mode
                if enable_memory_efficient and batch_timestamps < self.n_timestamps:
                    # 需要按时间戳分组处理charts_encoding
                    chart_indices_cpu = chart_indices.cpu()
                    timestamps_in_batch = torch.unique(chart_indices_cpu // self.n_charts_per_timestamp)
                    encodings_norm_sum = 0.
                    total_charts = 0
                    for t in timestamps_in_batch:
                        t = t.item()
                        timestamp_chart_mask = (chart_indices_cpu // self.n_charts_per_timestamp) == t
                        timestamp_chart_indices = chart_indices_cpu[timestamp_chart_mask]
                        if len(timestamp_chart_indices) > 0:
                            pts_uv_t = self._pts_uv[timestamp_chart_indices]
                            encodings_t = self.charts_encoding(pts_uv_t)
                            encodings_norm_sum += encodings_t.norm(dim=-1).sum()
                            total_charts += encodings_t.numel() // encodings_t.shape[-1]  # 总数除以最后一维
                    chart_encodings_norm_loss = encodings_norm_sum / total_charts if total_charts > 0 else 0.
                else:
                    # Process all charts (grouped by timestamp for safety)
                    encodings_norm_sum = 0.
                    total_charts = 0
                    for t in range(self.n_timestamps):
                        start_idx = t * self.n_charts_per_timestamp
                        end_idx = (t + 1) * self.n_charts_per_timestamp
                        pts_uv_t = self._pts_uv[start_idx:end_idx]
                        encodings_t = self.charts_encoding(pts_uv_t)
                        encodings_norm_sum += encodings_t.norm(dim=-1).sum()
                        total_charts += encodings_t.numel() // encodings_t.shape[-1]
                    chart_encodings_norm_loss = encodings_norm_sum / total_charts if total_charts > 0 else 0.
                loss += chart_encodings_norm_loss_weight * chart_encodings_norm_loss
                chart_encodings_norm_loss = chart_encodings_norm_loss.detach().item()

            if use_total_variation_on_depth_encodings:
                depth_encodings_tv_loss = (self.depth_encoding.encodings[..., 1:] - self.depth_encoding.encodings[..., :-1]).abs().mean()
                loss += total_variation_on_depth_encodings_weight * depth_encodings_tv_loss
                depth_encodings_tv_loss = depth_encodings_tv_loss.detach().item()
            backward_start = time.time()
            # Save total loss before dividing by accumulation_steps
            _total_loss = loss.detach().item()
            
            # Update parameters
            loss = loss / accumulation_steps
            loss.backward()
            backward_time = time.time() - backward_start
            if i_iter % 100 == 0:
                print("backward time: ", backward_time)
            # Optimization step
            if (i_iter + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Update learning rate scheduler
            if use_lr_scheduler and lr_scheduler is not None:
                lr_scheduler.step()
            
            #self.optimizer.zero_grad(set_to_none=True)
            
            # Log to TensorBoard
            if writer is not None:
                # Log main losses
                writer.add_scalar('Loss/Total', _total_loss, i_iter)
                writer.add_scalar('Loss/Depth', _loss, i_iter)
                
                # Log individual loss components (only if they were computed)
                if use_gradient_loss and 'grad_loss' in locals():
                    writer.add_scalar('Loss/Gradient', grad_loss, i_iter)
                if use_hessian_loss and 'hess_loss' in locals():
                    writer.add_scalar('Loss/Hessian', hess_loss, i_iter)
                if use_normal_loss and 'normal_loss' in locals():
                    writer.add_scalar('Loss/Normal', normal_loss, i_iter)
                if use_curvature_loss and 'curv_loss' in locals():
                    writer.add_scalar('Loss/Curvature', curv_loss, i_iter)
                if use_matching_loss and 'matching_loss' in locals():
                    writer.add_scalar('Loss/Matching', matching_loss, i_iter)
                if use_reprojection_loss and 'reprojection_loss' in locals():
                    writer.add_scalar('Loss/Reprojection', reprojection_loss, i_iter)
                if use_occlusion_loss and 'occlusion_loss' in locals():
                    writer.add_scalar('Loss/Occlusion', occlusion_loss, i_iter)
                if use_ssi_loss and 'ssi_loss_val' in locals():
                    writer.add_scalar('Loss/SSI', ssi_loss_val, i_iter)
                if regularize_chart_encodings_norms and 'chart_encodings_norm_loss' in locals():
                    writer.add_scalar('Loss/ChartEncodingsNorm', chart_encodings_norm_loss, i_iter)
                if use_total_variation_on_depth_encodings and 'depth_encodings_tv_loss' in locals():
                    writer.add_scalar('Loss/DepthEncodingsTV', depth_encodings_tv_loss, i_iter)
                
                # Log learning rates
                for param_group in self.optimizer.param_groups:
                    writer.add_scalar(f'LearningRate/{param_group["name"]}', param_group['lr'], i_iter)
                
                # Log timing information (optional, every 10 iterations)
                if i_iter % 10 == 0:
                    if 'backward_time' in locals():
                        writer.add_scalar('Timing/Backward', backward_time, i_iter)
                    if 'loss_computation_time' in locals():
                        writer.add_scalar('Timing/LossComputation', loss_computation_time, i_iter)
                    if 'gradient_loss_time' in locals():
                        writer.add_scalar('Timing/GradientLoss', gradient_loss_time, i_iter)
                    if 'normal_loss_time' in locals():
                        writer.add_scalar('Timing/NormalLoss', normal_loss_time, i_iter)
                    if 'matching_loss_time' in locals():
                        writer.add_scalar('Timing/MatchingLoss', matching_loss_time, i_iter)
                    if 'occlusion_loss_time' in locals():
                        writer.add_scalar('Timing/OcclusionLoss', occlusion_loss_time, i_iter)

            # Update matchings if needed
            if use_matching_loss and (matching_update_iters is not None) and (i_iter in matching_update_iters):
                if not (enable_memory_efficient and batch_timestamps < self.n_timestamps):
                    print("\n[INFO] Updating matchings for all timestamps.")
                    # Need to compute all deformed depths for matching update
                    with torch.no_grad():
                        _all_deformed_verts = self.verts
                        _all_deformed_depths_list = []
                        for t in range(self.n_timestamps):
                            start_idx = t * self.n_charts_per_timestamp
                            end_idx = (t + 1) * self.n_charts_per_timestamp
                            verts_t = _all_deformed_verts[start_idx:end_idx].reshape(self.n_charts_per_timestamp, -1, 3)
                            depths_t = self.cameras.p3d_cameras[list(range(start_idx, end_idx))].get_world_to_view_transform().transform_points(verts_t)[..., 2]
                            _all_deformed_depths_list.append(depths_t.reshape(self.n_charts_per_timestamp, self.pm_h, self.pm_w))
                        _all_deformed_depths = torch.cat(_all_deformed_depths_list, dim=0)

                        for t in range(self.n_timestamps):
                            chart_indices_t = list(range(t * self.n_charts_per_timestamp, (t + 1) * self.n_charts_per_timestamp))
                            timestamp_deformed_depths = _all_deformed_depths[chart_indices_t].detach()
                            matchers[t].update_references(reference_depths=timestamp_deformed_depths)
                            matchers[t].match(matching_thr)
                else:
                    if verbose:
                        print("\n[INFO] Matching update skipped in memory-efficient mode.")

            with torch.no_grad():
                iter_interval = n_iterations // 50
                if i_iter % 10 == 0:
                    iter_total_time = time.time() - iter_start_time
                    print(f"\n[TIMING SUMMARY] Iteration {i_iter} total time: {iter_total_time:.4f}s")
                    print("   > Module breakdown:")
                    if 'lr_update_time' in locals():
                        print(f"      > Learning rate update: {lr_update_time:.4f}s")
                    if 'loss_computation_time' in locals():
                        print(f"      > Loss computation: {loss_computation_time:.4f}s")
                    if 'gradient_loss_time' in locals():
                        print(f"      > Gradient loss: {gradient_loss_time:.4f}s")
                    if 'hessian_loss_time' in locals():
                        print(f"      > Hessian loss: {hessian_loss_time:.4f}s")
                    if 'normal_loss_time' in locals():
                        print(f"      > Normal loss: {normal_loss_time:.4f}s")
                    if 'curvature_loss_time' in locals():
                        print(f"      > Curvature loss: {curvature_loss_time:.4f}s")
                    if 'matching_loss_time' in locals():
                        print(f"      > Matching loss: {matching_loss_time:.4f}s")
                    if 'reprojection_loss_time' in locals():
                        print(f"      > Reprojection loss: {reprojection_loss_time:.4f}s")
                    if 'occlusion_loss_time' in locals():
                        print(f"      > Occlusion loss: {occlusion_loss_time:.4f}s")
                #verts_computation_time = time.time() - verts_computation_start
                    if 'verts_computation_time' in locals():
                        print(f"      > Verts computation: {verts_computation_time:.4f}s")
                    if 'depths_computation_time' in locals():
                        print(f"      > Depths computation: {depths_computation_time:.4f}s")
                    if 'data_transfer_time' in locals():
                        print(f"      > Data transfer: {data_transfer_time:.4f}s")
                    #if 'loss_computation_time' in locals():
                    #    print(f"      > Loss computation: {loss_computation_time:.4f}s")
                    if 'backward_time' in locals():
                        print(f"      > Backward: {backward_time:.4f}s")
                    #if 'optimizer_step_time' in locals():
                    #    print(f"      > Optimizer step: {optimizer_step_time:.4f}s")
                if i_iter % iter_interval == 0:
                    train_losses.append(_total_loss)
                    #print("x"*50)
                    #print(self._verts.mean(),self.verts.mean())
                    if True:
                        loss_dict = {"loss": f"{_total_loss:.5f}", "depth_loss": f"{_loss:.5f}"}
                        if use_gradient_loss:
                            loss_dict["grad"] = f"{grad_loss:.5f}"
                        if use_hessian_loss:
                            loss_dict["hess"] = f"{hess_loss:.5f}"
                        if use_normal_loss:
                            loss_dict["normal"] = f"{normal_loss:.5f}"
                        if use_curvature_loss:
                            loss_dict["curv"] = f"{curv_loss:.5f}"
                        if use_matching_loss:
                            loss_dict["matching"] = f"{matching_loss:.5f}"
                        if use_reprojection_loss:
                            loss_dict["reproj"] = f"{reprojection_loss:.5f}"
                        if use_occlusion_loss:
                            loss_dict["occlusion"] = f"{occlusion_loss:.5f}"
                        if use_ssi_loss:
                            loss_dict["ssi"] = f"{ssi_loss_val:.5f}"
                        if regularize_chart_encodings_norms:
                            loss_dict["ce_norm"] = f"{chart_encodings_norm_loss:.5f}"
                        if use_total_variation_on_depth_encodings:
                            loss_dict["de_tv"] = f"{depth_encodings_tv_loss:.5f}"
                        progress_bar.update(iter_interval)
                        progress_bar.set_postfix(loss_dict)
                    else:
                        if verbose:
                            print(f"Iteration {i_iter}: Loss = {_total_loss}")
                            print(f"   > Depth loss = {_loss}")
                            if use_gradient_loss:
                                print(f"   > Gradient Loss = {grad_loss}")
                            if use_hessian_loss:
                                print(f"   > Hessian Loss = {hess_loss}")
                            if use_normal_loss:
                                print(f"   > Normal Loss = {normal_loss}")
                            if use_curvature_loss:
                                print(f"   > Curvature Loss = {curv_loss}")
                            if use_matching_loss:
                                print(f"   > Matching Loss = {matching_loss}")
                            if use_reprojection_loss:
                                print(f"   > Reprojection Loss = {reprojection_loss}")
                            if use_occlusion_loss:
                                print(f"   > Occlusion Loss = {occlusion_loss}")
                            if regularize_chart_encodings_norms:
                                print(f"   > Chart encodings norms loss = {chart_encodings_norm_loss}")
                            if use_total_variation_on_depth_encodings:
                                print(f"   > Total variation on depth encodings loss = {depth_encodings_tv_loss}")

                            # Print timing summary every 10 iterations
                            

                            for name, param in self.named_parameters():
                                if param.requires_grad:
                                    print(f"      > {name}")
                                    print(f"         > Max: {param.max().item()}   Min: {param.min().item()}   Mean: {param.mean().item()}   Std: {param.std().item()}")
        with torch.no_grad():
            self._deformed_verts[...] = self.verts.clone()
        progress_bar.close()
        
        # Close TensorBoard writer
        if writer is not None:
            writer.close()
            if verbose:
                print(f"[INFO] TensorBoard logs saved to: {log_dir}")
        
        if verbose:
            print("Temporal optimization done.")
        self.train_losses = train_losses
