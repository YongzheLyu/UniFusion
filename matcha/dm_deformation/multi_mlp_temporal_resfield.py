import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from .multi_mlp import MultiLinear, initialize_multi_mlp_weights
from .resfield import Linear as ResFieldLinear


class MultiResFieldLinear(nn.Module):
    """Multi-head ResField Linear layer.
    
    Applies N ResField linear transformations to N batches of incoming data with temporal support.
    """
    def __init__(
        self, 
        n_heads: int,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        rank: int = 0,
        capacity: int = 0,
        mode: str = 'lookup',
        compression: str = 'vm',
        fuse_mode: str = 'add',
        coeff_ratio: float = 1.0,
        device=None, 
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        
        # Create ResFieldLinear layers for each head
        self.layers = nn.ModuleList([
            ResFieldLinear(
                in_features, out_features, bias=bias,
                rank=rank, capacity=capacity, mode=mode,
                compression=compression, fuse_mode=fuse_mode,
                coeff_ratio=coeff_ratio, **factory_kwargs
            ) for _ in range(n_heads)
        ])

    def forward(self, input: torch.Tensor, input_time=None, frame_id=None) -> torch.Tensor:
        """Forward pass with temporal parameters.
        
        Args:
            input: (n_heads, batch_size, in_features)
            input_time: (batch_size,) or (n_heads, batch_size) - continuous time values
            frame_id: (batch_size,) or (n_heads, batch_size) - discrete frame indices
        """
        outputs = []
        for i, layer in enumerate(self.layers):
            head_input = input[i]  # (batch_size, in_features)
            
            # Handle frame_id: if it's (n_heads,), extract the i-th element as a scalar
            # If it's (n_heads, batch_size), extract the i-th row
            # If it's (batch_size,), use it directly
            head_frame_id = None
            if frame_id is not None:
                if frame_id.dim() == 1:
                    if frame_id.shape[0] == self.n_heads:
                        # frame_id is (n_heads,), extract scalar for this head
                        # Convert to int to avoid indexing issues
                        head_frame_id = frame_id[i].item() if hasattr(frame_id[i], 'item') else int(frame_id[i])
                    else:
                        # frame_id is (batch_size,), use it directly
                        head_frame_id = frame_id
                elif frame_id.dim() == 2 and frame_id.shape[0] == self.n_heads:
                    # frame_id is (n_heads, batch_size), extract the i-th row
                    head_frame_id = frame_id[i]
                else:
                    # Use as is
                    head_frame_id = frame_id
            
            # Handle input_time similarly
            head_input_time = None
            if input_time is not None:
                if input_time.dim() == 1:
                    if input_time.shape[0] == self.n_heads:
                        head_input_time = input_time[i].item() if hasattr(input_time[i], 'item') else float(input_time[i])
                    else:
                        head_input_time = input_time
                elif input_time.dim() == 2 and input_time.shape[0] == self.n_heads:
                    head_input_time = input_time[i]
                else:
                    head_input_time = input_time
            
            head_output = layer(head_input, input_time=head_input_time, frame_id=head_frame_id)
            outputs.append(head_output)
        
        return torch.stack(outputs, dim=0)  # (n_heads, batch_size, out_features)


class DeformationMultiMLPResField(nn.Module):
    def __init__(
        self,
        n_heads,
        n_layer,
        layer_size,
        input_dim,
        output_dim,
        additional_input_dim=0,
        data_input_range_min=None,
        data_input_range_max=None,
        mlp_input_range_min=-1.,
        mlp_input_range_max=1.,
        output_range_min=-1.,
        output_range_max=1.,
        non_linearity=nn.ReLU(),
        final_non_linearity=None,
        positional_encoding=None,
        frequency_pos_encoding_freqs=4,
        output_points=False,  # if True, output dimension is set to 3 for point coordinates
        use_skip_connection=False,  # if True, add skip connection from initial input to last layer
        cam_param_dim=9,          # 9D camera params (6D rot + 3D T)
        cam_encoding_dim=32,      # encoded camera feature dimension
        use_cam_encoding=False,   # if True, encode cam_params -> cam_encoding_dim inside this MLP
        # ResFields parameters
        resfield_layers=None,  # List of layer indices to apply ResFields (None means all layers)
        resfield_rank=8,
        resfield_capacity=64,
        resfield_mode='lookup',
        resfield_compression='vm',
        resfield_fuse_mode='add',
        resfield_coeff_ratio=1.0,
        ):
        """Multi-head MLP with ResFields for temporal modeling.

        This uses ResFields Linear layers that can model temporal variations 
        in the network weights through low-rank residuals, using input_time and frame_id.

        Args:
            n_heads (_type_): Number of MLP heads (typically n_charts_per_timestamp)
            n_layer (_type_): Number of layers
            layer_size (_type_): Size of hidden layers
            input_dim (_type_): Dimension of spatial features
            output_dim (_type_): Dimension of output
            additional_input_dim (int, optional): Additional input dimension. Defaults to 0.
            data_input_range_min (_type_, optional): Input data range min. Defaults to None.
            data_input_range_max (_type_, optional): Input data range max. Defaults to None.
            mlp_input_range_min (_type_, optional): MLP input range min. Defaults to -1..
            mlp_input_range_max (_type_, optional): MLP input range max. Defaults to 1..
            output_range_min (_type_, optional): Output range min. Defaults to -1..
            output_range_max (_type_, optional): Output range max. Defaults to 1..
            non_linearity (_type_, optional): Nonlinearity to use. Defaults to nn.ReLU().
            final_non_linearity (_type_, optional): Final nonlinearity. Defaults to None.
            positional_encoding (_type_, optional): Positional encoding. Defaults to None.
            frequency_pos_encoding_freqs (int, optional): Frequency bands for pos encoding. Defaults to 4.
            output_points (bool, optional): If True, output 3D points. Defaults to False.
            use_skip_connection (bool, optional): If True, use skip connection. Defaults to False.
            cam_param_dim (int, optional): Camera parameter dimension. Defaults to 9.
            cam_encoding_dim (int, optional): Camera encoding dimension. Defaults to 32.
            use_cam_encoding (bool, optional): If True, encode camera params. Defaults to True.
            resfield_layers (list, optional): Layer indices to apply ResFields. Defaults to None (all layers).
            resfield_rank (int, optional): ResFields rank. Defaults to 8.
            resfield_capacity (int, optional): ResFields capacity. Defaults to 64.
            resfield_mode (str, optional): ResFields mode. Defaults to 'lookup'.
            resfield_compression (str, optional): ResFields compression. Defaults to 'vm'.
            resfield_fuse_mode (str, optional): ResFields fuse mode. Defaults to 'add'.
            resfield_coeff_ratio (float, optional): ResFields coefficient ratio. Defaults to 1.0.
        """
        super(DeformationMultiMLPResField, self).__init__()

        self.n_heads = n_heads
        self.n_layer = n_layer
        self.layer_size = layer_size
        self.non_linearity = non_linearity

        self.input_dim = input_dim
        self.original_output_dim = output_dim  # Save original output_dim
        self.output_points = output_points
        self.use_skip_connection = use_skip_connection
        
        # If output_points is True, set output_dim to 3
        if output_points:
            self.output_dim = 3
        else:
            self.output_dim = output_dim

        self.additional_input_dim = additional_input_dim

        # Camera encoding config
        self.cam_param_dim = cam_param_dim
        self.cam_encoding_dim = cam_encoding_dim
        self.use_cam_encoding = use_cam_encoding

        self.data_input_range_min = data_input_range_min
        self.data_input_range_max = data_input_range_max

        self.mlp_input_range_min = mlp_input_range_min
        self.mlp_input_range_max = mlp_input_range_max

        self.output_range_min = output_range_min
        self.output_range_max = output_range_max

        self.final_non_linearity = final_non_linearity

        self._positional_encoding = positional_encoding
        self.use_positional_encoding = positional_encoding is not None
        self.frequency_pos_encoding_freqs = frequency_pos_encoding_freqs

        # ResFields parameters
        self.resfield_layers = resfield_layers if resfield_layers is not None else list(range(n_layer))
        self.resfield_rank = resfield_rank
        self.resfield_capacity = resfield_capacity
        self.resfield_mode = resfield_mode
        self.resfield_compression = resfield_compression
        self.resfield_fuse_mode = resfield_fuse_mode
        self.resfield_coeff_ratio = resfield_coeff_ratio

        # Positional encoding
        if positional_encoding == 'frequency':
            from .encodings import FrequencyPositionalEncoding
            self.positional_encoding = FrequencyPositionalEncoding(input_dim, frequency_pos_encoding_freqs)
            first_layer_input_dim = additional_input_dim + input_dim * 2 * frequency_pos_encoding_freqs
        elif positional_encoding is None:
            first_layer_input_dim = additional_input_dim + input_dim
            print("No positional encoding.")
        else:
            raise ValueError("Unknown positional encoding.")

        # If we use camera encoding, make room for cam_encoding_dim in the first layer
        if self.use_cam_encoding and self.cam_encoding_dim > 0:
            first_layer_input_dim += self.cam_encoding_dim
           

        # MLP layers with optional skip connection to last layer
        # Save first_layer_input_dim for skip connection (if enabled)
        self.first_layer_input_dim = first_layer_input_dim

        # Camera encoding MLP: cam_param_dim -> cam_encoding_dim
        if self.use_cam_encoding and self.cam_param_dim > 0 and self.cam_encoding_dim > 0:
            self.cam_encoding = MultiLinear(n_heads, self.cam_param_dim, self.cam_encoding_dim)
        else:
            self.cam_encoding = None

        # Build MLP layers with ResFields
        if use_skip_connection:
            # Build MLP layers (excluding the last layer)
            layers = nn.ModuleList()
            # First layer
            layers.append(self._make_resfield_linear(0, first_layer_input_dim, layer_size))
            layers.append(non_linearity)
            
            # Hidden layers
            for i in range(1, n_layer-1):
                layers.append(self._make_resfield_linear(i, layer_size, layer_size))
                layers.append(non_linearity)
            
            # Store layers (excluding last layer) in Sequential
            self.mlp = nn.Sequential(*layers)
            
            # Last layer: input = layer_size (from previous layer) + first_layer_input_dim (skip connection)
            last_layer_input_dim = layer_size + first_layer_input_dim
            self.last_layer = self._make_resfield_linear(n_layer-1, last_layer_input_dim, self.output_dim)
        else:
            # Standard MLP without skip connection
            layers = nn.ModuleList()
            # First layer
            layers.append(self._make_resfield_linear(0, first_layer_input_dim, layer_size))
            layers.append(non_linearity)
            
            # Hidden layers
            for i in range(1, n_layer-1):
                layers.append(self._make_resfield_linear(i, layer_size, layer_size))
                layers.append(non_linearity)
            
            # Last layer
            layers.append(self._make_resfield_linear(n_layer-1, layer_size, self.output_dim))
            if final_non_linearity is not None:
                layers.append(final_non_linearity)
            
            self.mlp = nn.Sequential(*layers)
            self.last_layer = None  # Not used when skip connection is disabled

    def _make_resfield_linear(self, layer_idx, in_features, out_features):
        """Create a linear layer with or without ResFields based on configuration."""
        if layer_idx in self.resfield_layers:
            # Use ResFields for this layer
            return MultiResFieldLinear(
                self.n_heads, in_features, out_features,
                rank=self.resfield_rank,
                capacity=self.resfield_capacity,
                mode=self.resfield_mode,
                compression=self.resfield_compression,
                fuse_mode=self.resfield_fuse_mode,
                coeff_ratio=self.resfield_coeff_ratio
            )
        else:
            # Use standard MultiLinear for this layer
            return MultiLinear(self.n_heads, in_features, out_features)

    def forward(self, x, additional_input=None, cam_params=None, input_time=None, frame_id=None):
        """Forward pass with ResFields temporal modeling.

        Args:
            x: (n_heads, batch_size, input_dim) - Spatial features
            additional_input: (n_heads, batch_size, additional_input_dim) - Optional additional features
            cam_params: (n_heads, batch_size, cam_param_dim) - Optional camera parameters
            input_time: (batch_size,) or (n_heads, batch_size) - Continuous time values for ResFields
            frame_id: (batch_size,) or (n_heads, batch_size) - Discrete frame indices for ResFields

        Returns:
            output: (n_heads, batch_size, output_dim) - If output_points=True, output_dim=3 for point coordinates
        """
        # x should have shape (n_heads, batch_size, input_dim)
        if (additional_input is None) and self.additional_input_dim > 0:
            raise ValueError("Additional input is required.")

        # Rescale spatial input if needed (useless)
        if self.data_input_range_min is not None and self.data_input_range_max is not None:
            x_center = (self.data_input_range_max + self.data_input_range_min) / 2
            x_scale = (self.data_input_range_max - self.data_input_range_min) / 2
            res = (x - x_center) / x_scale

            input_center = (self.mlp_input_range_max + self.mlp_input_range_min) / 2
            input_scale = (self.mlp_input_range_max - self.mlp_input_range_min) / 2
            res = res * input_scale + input_center
        else:
            res = x

        # Apply positional encoding
        if self.use_positional_encoding:
            res = self.positional_encoding(res)

        # Concatenate additional input
        if additional_input is not None:
            res = torch.cat([res, additional_input], dim=-1)

        # Encode and concatenate camera parameters if enabled
        if self.use_cam_encoding:
            if cam_params is None:
                raise ValueError("cam_params must be provided when use_cam_encoding=True.")
            cam_feats = self.cam_encoding(cam_params)  # (n_heads, batch, cam_encoding_dim)
            res = torch.cat([res, cam_feats], dim=-1)

        if self.use_skip_connection:
            # Save initial input for skip connection (before MLP processing)
            initial_input = res.clone()

            # Apply MLP (all layers except the last one)
            res = self._forward_mlp_with_temporal(res, input_time=input_time, frame_id=frame_id)

            # Skip connection: concatenate initial input to the last layer input
            res = torch.cat([res, initial_input], dim=-1)

            # Apply last layer with skip connection
            if isinstance(self.last_layer, MultiResFieldLinear):
                res = self.last_layer(res, input_time=input_time, frame_id=frame_id)
            else:
                res = self.last_layer(res)

            # Apply final nonlinearity if specified
            if self.final_non_linearity is not None:
                res = self.final_non_linearity(res)
        else:
            # Standard forward pass without skip connection
            res = self._forward_mlp_with_temporal(res, input_time=input_time, frame_id=frame_id)

        # Rescale output if needed
        if self.output_range_min is not None and self.output_range_max is not None:
            output_center = (self.output_range_max + self.output_range_min) / 2
            output_scale = (self.output_range_max - self.output_range_min) / 2
            res = res * output_scale + output_center

        return res

    def _forward_mlp_with_temporal(self, x, input_time=None, frame_id=None):
        """Forward through MLP layers with temporal support."""
        for layer in self.mlp:
            if isinstance(layer, MultiResFieldLinear):
                x = layer(x, input_time=input_time, frame_id=frame_id)
            else:
                x = layer(x)
        return x


class DeformationMLPResField(nn.Module):
    def __init__(
        self,
        n_layer,
        layer_size,
        input_dim,
        output_dim,
        additional_input_dim=0,
        data_input_range_min=None,
        data_input_range_max=None,
        mlp_input_range_min=-1.,
        mlp_input_range_max=1.,
        output_range_min=-1.,
        output_range_max=1.,
        non_linearity=nn.ReLU(),
        final_non_linearity=None,
        positional_encoding=None,
        frequency_pos_encoding_freqs=4,
        # ResFields parameters
        resfield_layers=None,  # List of layer indices to apply ResFields (None means all layers)
        resfield_rank=8,
        resfield_capacity=64,
        resfield_mode='lookup',
        resfield_compression='vm',
        resfield_fuse_mode='add',
        resfield_coeff_ratio=1.0,
        ):
        """Single-head MLP with ResFields for temporal modeling.

        Single-head version of DeformationMultiMLPResField. Uses standard ResFieldLinear instead
        of MultiResFieldLinear.

        Args:
            n_layer (_type_): Number of layers in the MLP
            layer_size (_type_): Size of each hidden layer
            input_dim (_type_): Dimension of spatial features
            output_dim (_type_): Dimension of output
            additional_input_dim (int, optional): Dimension of additional input features. Defaults to 0.
            data_input_range_min (_type_, optional): Minimum value of input data range for rescaling. Defaults to None.
            data_input_range_max (_type_, optional): Maximum value of input data range for rescaling. Defaults to None.
            mlp_input_range_min (_type_, optional): Minimum value of MLP input range. Defaults to -1..
            mlp_input_range_max (_type_, optional): Maximum value of MLP input range. Defaults to 1..
            output_range_min (_type_, optional): Minimum value of output range for rescaling. Defaults to -1..
            output_range_max (_type_, optional): Maximum value of output range for rescaling. Defaults to 1..
            non_linearity (_type_, optional): Nonlinearity to use in the MLP. Defaults to nn.ReLU().
            final_non_linearity (_type_, optional): If None, no nonlinearity is applied after the last layer. Defaults to None.
            positional_encoding (_type_, optional): Positional encoding to use on the spatial input.
                If None, no positional encoding is used. Defaults to None.
            frequency_pos_encoding_freqs (int, optional): Number of frequency bands for positional encoding. Defaults to 4.
            resfield_layers (list, optional): Layer indices to apply ResFields. Defaults to None (all layers).
            resfield_rank (int, optional): ResFields rank. Defaults to 8.
            resfield_capacity (int, optional): ResFields capacity. Defaults to 64.
            resfield_mode (str, optional): ResFields mode. Defaults to 'lookup'.
            resfield_compression (str, optional): ResFields compression. Defaults to 'vm'.
            resfield_fuse_mode (str, optional): ResFields fuse mode. Defaults to 'add'.
            resfield_coeff_ratio (float, optional): ResFields coefficient ratio. Defaults to 1.0.
        """
        super(DeformationMLPResField, self).__init__()

        self.n_layer = n_layer
        self.layer_size = layer_size
        self.non_linearity = non_linearity

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.additional_input_dim = additional_input_dim

        self.data_input_range_min = data_input_range_min
        self.data_input_range_max = data_input_range_max

        self.mlp_input_range_min = mlp_input_range_min
        self.mlp_input_range_max = mlp_input_range_max

        self.output_range_min = output_range_min
        self.output_range_max = output_range_max

        self.final_non_linearity = final_non_linearity

        self._positional_encoding = positional_encoding
        self.use_positional_encoding = positional_encoding is not None
        self.frequency_pos_encoding_freqs = frequency_pos_encoding_freqs

        # ResFields parameters
        self.resfield_layers = resfield_layers if resfield_layers is not None else list(range(n_layer))
        self.resfield_rank = resfield_rank
        self.resfield_capacity = resfield_capacity
        self.resfield_mode = resfield_mode
        self.resfield_compression = resfield_compression
        self.resfield_fuse_mode = resfield_fuse_mode
        self.resfield_coeff_ratio = resfield_coeff_ratio

        # Positional encoding
        if positional_encoding == 'frequency':
            from .encodings import FrequencyPositionalEncoding
            self.positional_encoding = FrequencyPositionalEncoding(input_dim, frequency_pos_encoding_freqs)
            first_layer_input_dim = additional_input_dim + input_dim * 2 * frequency_pos_encoding_freqs
        elif positional_encoding is None:
            first_layer_input_dim = additional_input_dim + input_dim
            print("No positional encoding.")
        else:
            raise ValueError("Unknown positional encoding.")

        # MLP layers with ResFields
        layers = nn.ModuleList()
        
        # First layer
        layers.append(self._make_resfield_linear(0, first_layer_input_dim, layer_size))
        layers.append(non_linearity)
        
        # Hidden layers
        for i in range(1, n_layer-1):
            layers.append(self._make_resfield_linear(i, layer_size, layer_size))
            layers.append(non_linearity)
        
        # Last layer
        layers.append(self._make_resfield_linear(n_layer-1, layer_size, output_dim))
        if final_non_linearity is not None:
            layers.append(final_non_linearity)
        
        self.mlp = nn.Sequential(*layers)

    def _make_resfield_linear(self, layer_idx, in_features, out_features):
        """Create a linear layer with or without ResFields based on configuration."""
        if layer_idx in self.resfield_layers:
            # Use ResFields for this layer
            return ResFieldLinear(
                in_features, out_features,
                rank=self.resfield_rank,
                capacity=self.resfield_capacity,
                mode=self.resfield_mode,
                compression=self.resfield_compression,
                fuse_mode=self.resfield_fuse_mode,
                coeff_ratio=self.resfield_coeff_ratio
            )
        else:
            # Use standard Linear for this layer
            return nn.Linear(in_features, out_features)

    def forward(self, x, additional_input=None, input_time=None, frame_id=None):
        """Forward pass with ResFields temporal modeling.

        Args:
            x: (batch_size, input_dim) - Spatial features
            additional_input: (batch_size, additional_input_dim) - Optional additional features
            input_time: (batch_size,) - Continuous time values for ResFields
            frame_id: (batch_size,) - Discrete frame indices for ResFields

        Returns:
            output: (batch_size, output_dim)
        """
        # 如果输入还是 num_heads,batch,input_dim 的形状，直接把num_heads reshape到batch_size维度
        # 检查 x 的形状（假定 x 是 (num_heads, batch, input_dim)）
        if x.dim() == 3 and x.shape[0] < x.shape[1]:
            # 认为第一个维度是 num_heads，合并到 batch 维
            num_heads, batch, input_dim = x.shape
            x = x.reshape(num_heads * batch, input_dim)
            # 同步 additional_input
            if additional_input is not None and additional_input.shape[0] == num_heads and additional_input.shape[1] == batch:
                _, _, additional_input_dim = additional_input.shape
                additional_input = additional_input.reshape(num_heads * batch, additional_input_dim)
        
        # x should have shape (batch_size, input_dim)
        if (additional_input is None) and self.additional_input_dim > 0:
            raise ValueError("Additional input is required.")

        # Rescale spatial input if needed
        if self.data_input_range_min is not None and self.data_input_range_max is not None:
            x_center = (self.data_input_range_max + self.data_input_range_min) / 2
            x_scale = (self.data_input_range_max - self.data_input_range_min) / 2
            res = (x - x_center) / x_scale

            input_center = (self.mlp_input_range_max + self.mlp_input_range_min) / 2
            input_scale = (self.mlp_input_range_max - self.mlp_input_range_min) / 2
            res = res * input_scale + input_center
        else:
            res = x

        # Apply positional encoding
        if self.use_positional_encoding:
            res = self.positional_encoding(res)

        # Concatenate additional input
        if additional_input is not None:
            res = torch.cat([res, additional_input], dim=-1)

        # Apply MLP with temporal support
        res = self._forward_mlp_with_temporal(res, input_time=input_time, frame_id=frame_id)

        # Rescale output if needed
        if self.output_range_min is not None and self.output_range_max is not None:
            output_center = (self.output_range_max + self.output_range_min) / 2
            output_scale = (self.output_range_max - self.output_range_min) / 2
            res = res * output_scale + output_center
        
        # 如果输入最初是 (num_heads, batch, input_dim) 这样的形状，则输出也reshape回 (num_heads, batch, output_dim)
        if 'num_heads' in locals() and 'batch' in locals() and res.shape[0] == num_heads * batch:
            output_dim = res.shape[1]
            res = res.reshape(num_heads, batch, output_dim)
        
        return res

    def _forward_mlp_with_temporal(self, x, input_time=None, frame_id=None):
        """Forward through MLP layers with temporal support."""
        for layer in self.mlp:
            if isinstance(layer, ResFieldLinear):
                x = layer(x, input_time=input_time, frame_id=frame_id)
            else:
                x = layer(x)
        return x
