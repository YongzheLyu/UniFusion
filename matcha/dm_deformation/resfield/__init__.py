"""
ResFields: Residual Neural Fields for Spatiotemporal Signals

Based on ResFields (ICLR 2024) by Mihajlovic et al.
Paper: https://arxiv.org/abs/2309.03160
Original repository: https://github.com/markomih/ResFields

This module provides time-dependent residual weights for MLP layers,
enabling efficient modeling of temporal dynamics in neural fields.
"""

from .linear import Linear

__all__ = ['Linear']