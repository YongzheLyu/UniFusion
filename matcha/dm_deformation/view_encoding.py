import numpy as np
import torch
import torch.nn as nn
from typing import List


class ViewEncoding(nn.Module):
    """View encoding for multi-camera optimization.

    This module creates view features that can be concatenated to spatial features
    in the MLP. Supports two modes:
    - 'learned': Learnable embeddings for each view/camera
    - 'positional': Sinusoidal positional encoding based on view/camera index
    """
    def __init__(
        self,
        n_views:int,  # n_charts_per_timestamp, number of cameras/views
        encoding_dim:int=32,
        encoding_type:str='learned',  # 'learned' or 'positional'
        max_freq:int=8,  # Only used for positional encoding
        initialization_range:float=1e-2,
    ):
        """Initialize view encoding.

        Args:
            n_views (int): Number of views/cameras (n_charts_per_timestamp)
            encoding_dim (int, optional): Dimension of view features. Defaults to 32.
            encoding_type (str, optional): Type of encoding ('learned' or 'positional'). Defaults to 'learned'.
            max_freq (int, optional): Maximum frequency for positional encoding. Defaults to 8.
            initialization_range (float, optional): Initialization range for learned embeddings. Defaults to 1e-2.
        """
        super(ViewEncoding, self).__init__()
        self.n_views = n_views
        self.encoding_dim = encoding_dim
        self.encoding_type = encoding_type
        self.initialization_range = initialization_range

        if encoding_type == 'learned':
            # Learnable view embeddings
            self.view_embeddings = nn.Parameter(
                initialization_range * (-1. + 2. * torch.rand(n_views, encoding_dim)),
                requires_grad=True,
            )
        elif encoding_type == 'positional':
            # Sinusoidal positional encoding
            self.max_freq = max_freq
            # Pre-compute frequencies
            freqs = 2 ** torch.linspace(0, max_freq, encoding_dim // 2)
            self.register_buffer('freqs', freqs)
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}. Must be 'learned' or 'positional'.")

    def forward(self, view_indices:torch.Tensor):
        """Get view features for given view/camera indices.

        Args:
            view_indices (torch.Tensor): Shape (n_charts,) - View/camera index for each chart

        Returns:
            torch.Tensor: Shape (n_charts, encoding_dim) - View features
        """
        if self.encoding_type == 'learned':
            return self.view_embeddings[view_indices]

        elif self.encoding_type == 'positional':
            # Normalize views to [0, 1]
            v_normalized = view_indices.float() / max(1, self.n_views - 1)

            # Sinusoidal encoding
            angles = v_normalized[:, None] * self.freqs[None, :]  # (n_charts, encoding_dim//2)
            encodings = torch.cat([
                torch.sin(2 * np.pi * angles),
                torch.cos(2 * np.pi * angles)
            ], dim=-1)  # (n_charts, encoding_dim)

            return encodings

