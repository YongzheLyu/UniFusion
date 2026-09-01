import numpy as np
import torch
import torch.nn as nn
from typing import List


class TemporalEncoding(nn.Module):
    """Temporal encoding for multi-timestamp optimization.

    This module creates temporal features that can be concatenated to spatial features
    in the MLP. Supports two modes:
    - 'learned': Learnable embeddings for each timestamp
    - 'positional': Sinusoidal positional encoding based on timestamp index
    """
    def __init__(
        self,
        n_timestamps:int,
        encoding_dim:int=8,
        encoding_type:str='learned',  # 'learned' or 'positional'
        max_freq:int=5,  # Only used for positional encoding
        initialization_range:float=1e-2,
    ):
        """Initialize temporal encoding.

        Args:
            n_timestamps (int): Number of timestamps in the sequence
            encoding_dim (int, optional): Dimension of temporal features. Defaults to 8.
            encoding_type (str, optional): Type of encoding ('learned' or 'positional'). Defaults to 'learned'.
            max_freq (int, optional): Maximum frequency for positional encoding. Defaults to 5.
            initialization_range (float, optional): Initialization range for learned embeddings. Defaults to 1e-2.
        """
        super(TemporalEncoding, self).__init__()
        self.n_timestamps = n_timestamps
        self.encoding_dim = encoding_dim
        self.encoding_type = encoding_type
        self.initialization_range = initialization_range

        if encoding_type == 'learned':
            # Learnable temporal embeddings
            self.time_embeddings = nn.Parameter(
                initialization_range * (-1. + 2. * torch.rand(n_timestamps, encoding_dim)),
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

    def forward(self, timestamp_indices:torch.Tensor):
        """Get temporal features for given timestamp indices.

        Args:
            timestamp_indices (torch.Tensor): Shape (n_charts,) - Timestamp index for each chart

        Returns:
            torch.Tensor: Shape (n_charts, encoding_dim) - Temporal features
        """
        if self.encoding_type == 'learned':
            return self.time_embeddings[timestamp_indices]

        elif self.encoding_type == 'positional':
            # Normalize timestamps to [0, 1]
            t_normalized = timestamp_indices.float() / max(1, self.n_timestamps - 1)

            # Sinusoidal encoding
            angles = t_normalized[:, None] * self.freqs[None, :]  # (n_charts, encoding_dim//2)
            encodings = torch.cat([
                torch.sin(2 * np.pi * angles),
                torch.cos(2 * np.pi * angles)
            ], dim=-1)  # (n_charts, encoding_dim)

            return encodings
