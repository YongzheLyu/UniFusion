import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from .multi_mlp import MultiLinear, initialize_multi_mlp_weights


class DeformationMultiMLPTemporal(nn.Module):
    def __init__(
        self,
        n_heads,
        n_layer,
        layer_size,
        input_dim,
        output_dim,
        time_feature_dim=8,  # NEW: dimension for time features
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
        output_points=False,  # NEW: if True, output dimension is set to 3 for point coordinates
        use_skip_connection=False,  # NEW: if True, add skip connection from initial input to last layer
        cam_param_dim=9,          # NEW: 9D camera params (6D rot + 3D T)
        cam_encoding_dim=32,      # NEW: encoded camera feature dimension
        use_cam_encoding=True,   # NEW: if True, encode cam_params -> cam_encoding_dim inside this MLP
        add_time_features=False,  # NEW: if True, add time features to the spatial features instead of concatenating
        ):
        """Multi-head MLP with temporal feature support.

        The key difference from DeformationMultiMLP is that this version accepts
        an additional time_features input which is concatenated to the spatial features.

        Args:
            n_heads (_type_): Number of MLP heads (typically n_charts_per_timestamp)
            n_layer (_type_): _description_
            layer_size (_type_): _description_
            input_dim (_type_): Dimension of spatial features
            output_dim (_type_): _description_
            time_feature_dim (int, optional): Dimension of temporal features. Defaults to 8.
            additional_input_dim (int, optional): _description_. Defaults to 0.
            data_input_range_min (_type_, optional): _description_. Defaults to None.
            data_input_range_max (_type_, optional): _description_. Defaults to None.
            mlp_input_range_min (_type_, optional): _description_. Defaults to -1..
            mlp_input_range_max (_type_, optional): _description_. Defaults to 1..
            output_range_min (_type_, optional): _description_. Defaults to -1..
            output_range_max (_type_, optional): _description_. Defaults to 1..
            non_linearity (_type_, optional): Nonlinearity to use in the MLP. Defaults to nn.ReLU().
            final_non_linearity (_type_, optional): If None, no nonlinearity is applied after the last layer. Defaults to None.
            positional_encoding (_type_, optional): Positional encoding to use on the spatial input.
                If None, no positional encoding is used. Defaults to None.
            frequency_pos_encoding_freqs (int, optional): _description_. Defaults to 4.
            output_points (bool, optional): If True, the last layer output dimension is set to 3 for point coordinates. Defaults to False.
            use_skip_connection (bool, optional): If True, concatenate initial input features to the last layer input. Defaults to False.
        """
        super(DeformationMultiMLPTemporal, self).__init__()

        self.n_heads = n_heads
        self.n_layer = n_layer
        self.layer_size = layer_size
        self.non_linearity = non_linearity

        self.input_dim = input_dim
        self.original_output_dim = output_dim  # Save original output_dim
        self.output_points = output_points
        self.use_skip_connection = use_skip_connection
        self.add_time_features = add_time_features
        # If output_points is True, set output_dim to 3
        if output_points:
            self.output_dim = 3
        else:
            self.output_dim = output_dim
            
        self.time_feature_dim = time_feature_dim

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

        # Positional encoding (not used in current implementation, but kept for compatibility)
        if positional_encoding == 'frequency':
            from .encodings import FrequencyPositionalEncoding
            self.positional_encoding = FrequencyPositionalEncoding(input_dim, frequency_pos_encoding_freqs)
            first_layer_input_dim = additional_input_dim + time_feature_dim + input_dim * 2 * frequency_pos_encoding_freqs
        elif positional_encoding is None:
            # First layer input = spatial features + time features + additional features
            first_layer_input_dim = additional_input_dim + time_feature_dim + input_dim
            if self.add_time_features:
                first_layer_input_dim = time_feature_dim
            print("No positional encoding.")
        else:
            raise ValueError("Unknown positional encoding.")

        # If we use camera encoding, make room for cam_encoding_dim in the first layer
        if self.use_cam_encoding and self.cam_encoding_dim > 0:
            if not self.add_time_features:
                first_layer_input_dim += self.cam_encoding_dim
           

        # MLP layers with optional skip connection to last layer
        # Save first_layer_input_dim for skip connection (if enabled)
        self.first_layer_input_dim = first_layer_input_dim

        # Camera encoding MLP: cam_param_dim -> cam_encoding_dim
        if self.use_cam_encoding and self.cam_param_dim > 0 and self.cam_encoding_dim > 0:
            self.cam_encoding = MultiLinear(n_heads, self.cam_param_dim, self.cam_encoding_dim)
        else:
            self.cam_encoding = None

        if use_skip_connection:
            # Build MLP layers (excluding the last layer)
            layers = nn.ModuleList()
            layers.append(MultiLinear(n_heads, first_layer_input_dim, layer_size))
            layers.append(non_linearity)
            for i in range(n_layer-2):
                layers.append(MultiLinear(n_heads, layer_size, layer_size))
                layers.append(non_linearity)
            
            # Last layer: input = layer_size (from previous layer) + first_layer_input_dim (skip connection)
            last_layer_input_dim = layer_size + first_layer_input_dim
            self.last_layer = MultiLinear(n_heads, last_layer_input_dim, self.output_dim)
            
            # Store layers (excluding last layer) in Sequential
            self.mlp = nn.Sequential(*layers)
        else:
            # Standard MLP without skip connection
            layers = nn.ModuleList()
            layers.append(MultiLinear(n_heads, first_layer_input_dim, layer_size))
            layers.append(non_linearity)
            for i in range(n_layer-2):
                layers.append(MultiLinear(n_heads, layer_size, layer_size))
                layers.append(non_linearity)
            layers.append(MultiLinear(n_heads, layer_size, self.output_dim))
            if final_non_linearity is not None:
                layers.append(final_non_linearity)
            self.mlp = nn.Sequential(*layers)
            self.last_layer = None  # Not used when skip connection is disabled
        
        # Note: final_non_linearity is already stored in self.final_non_linearity (line 87)

    def forward(self, x, time_features, additional_input=None, cam_params=None):
        """Forward pass with temporal features.

        Args:
            x: (n_heads, batch_size, input_dim) - Spatial features
            time_features: (n_heads, batch_size, time_feature_dim) - Temporal features
            additional_input: (n_heads, batch_size, additional_input_dim) - Optional additional features

        Returns:
            output: (n_heads, batch_size, output_dim) - If output_points=True, output_dim=3 for point coordinates
        """
        # x should have shape (n_heads, batch_size, input_dim)
        # time_features should have shape (n_heads, batch_size, time_feature_dim)
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

        # Concatenate time features
        # Ensure time_features is on the same device as res
        time_features = time_features.to(res.device)
        #res = torch.cat([res, time_features], dim=-1)
        if self.add_time_features:
            res = res + time_features
        else:
            res = torch.cat([res, time_features], dim=-1)

        # Concatenate additional input
        if additional_input is not None:
            additional_input = additional_input.to(res.device)
            res = torch.cat([res, additional_input], dim=-1)

        # Encode and concatenate camera parameters if enabled
        if self.use_cam_encoding:
            if cam_params is None:
                raise ValueError("cam_params must be provided when use_cam_encoding=True.")
            cam_params = cam_params.to(res.device)
            cam_feats = self.cam_encoding(cam_params)  # (n_heads, batch, cam_encoding_dim)
            if self.add_time_features:
                res = res + cam_feats
            else:   
                res = torch.cat([res, cam_feats], dim=-1)

        if self.use_skip_connection:
            # Save initial input for skip connection (before MLP processing)
            initial_input = res.clone()

            # Apply MLP (all layers except the last one)
            res = self.mlp(res)

            # Skip connection: concatenate initial input to the last layer input
            res = torch.cat([res, initial_input], dim=-1)

            # Apply last layer with skip connection
            res = self.last_layer(res)

            # Apply final nonlinearity if specified
            if self.final_non_linearity is not None:
                res = self.final_non_linearity(res)
        else:
            # Standard forward pass without skip connection
            res = self.mlp(res)

        # Rescale output if needed
        if self.output_range_min is not None and self.output_range_max is not None:
            output_center = (self.output_range_max + self.output_range_min) / 2
            output_scale = (self.output_range_max - self.output_range_min) / 2
            res = res * output_scale + output_center

        return res


class DeformationMLPTemporal(nn.Module):
    def __init__(
        self,
        n_layer,
        layer_size,
        input_dim,
        output_dim,
        time_feature_dim=8,  # NEW: dimension for time features
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
        ):
        """Single-head MLP with temporal feature support.

        Single-head version of DeformationMultiMLPTemporal. The key difference is that
        this version does not have the n_heads dimension, using standard nn.Linear instead
        of MultiLinear.

        Args:
            n_layer (_type_): Number of layers in the MLP
            layer_size (_type_): Size of each hidden layer
            input_dim (_type_): Dimension of spatial features
            output_dim (_type_): Dimension of output
            time_feature_dim (int, optional): Dimension of temporal features. Defaults to 8.
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
        """
        super(DeformationMLPTemporal, self).__init__()

        self.n_layer = n_layer
        self.layer_size = layer_size
        self.non_linearity = non_linearity

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_feature_dim = time_feature_dim

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

        # Positional encoding (not used in current implementation, but kept for compatibility)
        if positional_encoding == 'frequency':
            from .encodings import FrequencyPositionalEncoding
            self.positional_encoding = FrequencyPositionalEncoding(input_dim, frequency_pos_encoding_freqs)
            first_layer_input_dim = additional_input_dim + time_feature_dim + input_dim * 2 * frequency_pos_encoding_freqs
        elif positional_encoding is None:
            # First layer input = spatial features + time features + additional features
            first_layer_input_dim = additional_input_dim + time_feature_dim + input_dim
            print("No positional encoding.")
        else:
            raise ValueError("Unknown positional encoding.")

        # MLP layers
        layers = nn.ModuleList()
        layers.append(nn.Linear(first_layer_input_dim, layer_size))
        layers.append(non_linearity)
        for i in range(n_layer-2):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(non_linearity)
        layers.append(nn.Linear(layer_size, output_dim))
        if final_non_linearity is not None:
            layers.append(final_non_linearity)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, time_features, additional_input=None):
        """Forward pass with temporal features.

        Args:
            x: (batch_size, input_dim) - Spatial features
            time_features: (batch_size, time_feature_dim) - Temporal features
            additional_input: (batch_size, additional_input_dim) - Optional additional features

        Returns:
            output: (batch_size, output_dim)
        """
        # 如果输入还是 num_heads,batch,input_dim 的形状，直接把num_heads reshape到batch_size维度
        # 检查 x 的形状（假定 x 是 (num_heads, batch, input_dim)）
        if x.dim() == 3 and x.shape[0] < x.shape[1]:
            # 认为第一个维度是 num_heads，合并到 batch 维
            num_heads, batch, input_dim = x.shape
            x = x.reshape(num_heads * batch, input_dim)
            # 同步 time_features 和 additional_input
            if time_features is not None and time_features.shape[0] == num_heads and time_features.shape[1] == batch:
                _, _, time_feature_dim = time_features.shape
                time_features = time_features.reshape(num_heads * batch, time_feature_dim)
            if additional_input is not None and additional_input.shape[0] == num_heads and additional_input.shape[1] == batch:
                _, _, additional_input_dim = additional_input.shape
                additional_input = additional_input.reshape(num_heads * batch, additional_input_dim)
        # x should have shape (batch_size, input_dim)
        # time_features should have shape (batch_size, time_feature_dim)
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

        # Concatenate time features
        # Ensure time_features is on the same device as res
        time_features = time_features.to(res.device)
        res = torch.cat([res, time_features], dim=-1)

        # Concatenate additional input
        if additional_input is not None:
            additional_input = additional_input.to(res.device)
            res = torch.cat([res, additional_input], dim=-1)

        # Apply MLP
        res = self.mlp(res)

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