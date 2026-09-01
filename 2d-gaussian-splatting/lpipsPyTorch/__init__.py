import torch

from .modules.lpips import LPIPS

# Global cache for LPIPS models to avoid repeated initialization
_lpips_cache = {}


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    cache_key = f"{net_type}_{version}_{device}"

    if cache_key not in _lpips_cache:
        _lpips_cache[cache_key] = LPIPS(net_type, version).to(device).eval()

    criterion = _lpips_cache[cache_key]
    return criterion(x, y)
