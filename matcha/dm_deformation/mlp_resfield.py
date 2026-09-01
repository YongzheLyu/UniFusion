import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from resfield import Linear as ResFieldLinear
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

class ResFieldMLP(nn.Module):
    """MLP with ResFields time-dependent weights.

    This extends VanillaMLP with ResFields Linear layers that can model
    temporal variations in the network weights through low-rank residuals.
    """
    def __init__(self, dim_in, dim_out, config):
        super().__init__()

        self.n_neurons = config['n_neurons']
        self.n_hidden_layers = config['n_hidden_layers']
        self.sphere_init = config.get('sphere_init', False)
        self.weight_norm = config.get('weight_norm', False)
        self.inverse_sphere_init = config.get('inverse_sphere_init', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)

        # ResFields parameters
        resfield_layers = config.get('resfield_layers', [])
        resfield_rank = config.get('resfield_rank', 0)
        resfield_capacity = config.get('resfield_capacity', 0)
        resfield_mode = config.get('resfield_mode', 'lookup')
        resfield_compression = config.get('resfield_compression', 'vm')
        resfield_fuse_mode = config.get('resfield_fuse_mode', 'add')
        resfield_coeff_ratio = config.get('resfield_coeff_ratio', 1.0)

        self.layers = []
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        for i in range(len(dims) - 1):
            is_first = (i == 0)
            is_last = (i == len(dims) - 2)

            # Use ResFields Linear for ALL layers (with rank=0 for non-temporal layers)
            # This matches the ResFields pattern exactly
            _rank = resfield_rank if i in resfield_layers else 0
            _capacity = resfield_capacity if i in resfield_layers else 0
            layer = ResFieldLinear(
                dims[i], dims[i+1],
                rank=_rank,
                capacity=_capacity,
                mode=resfield_mode,
                compression=resfield_compression,
                fuse_mode=resfield_fuse_mode,
                coeff_ratio=resfield_coeff_ratio
            )

            # Apply initialization
            if self.sphere_init:
                if is_last:
                    if self.inverse_sphere_init:
                        torch.nn.init.constant_(layer.bias, self.sphere_init_radius)
                        torch.nn.init.normal_(layer.weight, mean=-math.sqrt(math.pi) / math.sqrt(dims[i]), std=0.0001)
                    else:
                        torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                        torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dims[i]), std=0.0001)
                elif is_first:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dims[i+1]))
                else:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dims[i+1]))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

            if self.weight_norm:
                layer = nn.utils.weight_norm(layer)

            self.layers.append(layer)
            if not is_last:
                self.layers.append(self.make_activation())

        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config.get('output_activation', 'none'))

    def forward(self, x, *args, frame_id=None, input_time=None, **kwargs):
        """Forward pass with optional temporal parameters.

        Args:
            x: Input tensor
            frame_id: Frame index for ResFields layers (discrete frames)
            input_time: Continuous time value for ResFields layers (range: -1 to 1)
        """
        for layer in self.layers:
            if isinstance(layer, ResFieldLinear):
                # All linear layers are ResFieldLinear - pass temporal parameters
                x = layer(x.float(), input_time=input_time, frame_id=frame_id)
            else:
                # Activation or other layers
                x = layer(x)
        x = self.output_activation(x)
        return x

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)
        
class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))
trunc_exp = _TruncExp.apply
def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name == 'trunc_exp':
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name == 'tanh':
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)