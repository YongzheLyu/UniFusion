"""
ResFields Linear Layer

Time-dependent linear transformation: y = x(A + δA_t)^T + b
where δA_t is a low-rank residual weight matrix that varies with time/frame.

Original implementation from ResFields (ICLR 2024)
"""

import torch

try:
    import tensorly as tl
    from tensorly.random.base import random_cp
    tl.set_backend('pytorch')
except ImportError:
    # tensorly is optional, only needed for CP/Tucker compression modes
    pass


class Linear(torch.nn.Linear):
    r"""Applies a ResField Linear transformation to the incoming data: :math:`y = x(A + \delta A_t)^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        rank: value for the the low rank decomposition (0 = disabled/standard linear layer)
        capacity: size of the temporal dimension (0 = disabled/standard linear layer)
        mode: 'lookup' (discrete frames) or 'interpolation' (continuous time)
        compression: compression type - 'vm', 'cp', 'tucker', 'none', 'resnet', 'vm_noweight', 'vm_attention', 'loe'
        fuse_mode: how to combine residual with base weight - 'add', 'mul', or 'none'
        coeff_ratio: coefficient ratio for capacity adjustment

    Attributes:
        weight: (F_out x F_in) base weight matrix
        bias: the learnable bias of the module of shape :math:`(\text{out\_features})`.

    Examples::

        >>> m = Linear(20, 30, rank=10, capacity=100)
        >>> input_x, frame_id = torch.randn(128, 20), torch.tensor([5])
        >>> output = m(input_x, frame_id=frame_id)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, rank=None, capacity=None, mode='lookup',
                 compression='vm', fuse_mode='add', coeff_ratio=1.0) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert mode in ['lookup', 'interpolation', 'cp']
        assert compression in ['vm', 'cp', 'none', 'tucker', 'resnet', 'vm_noweight', 'vm_attention', 'loe']
        assert fuse_mode in ['add', 'mul', 'none']
        self.rank = rank
        self.fuse_mode = fuse_mode
        self.capacity = capacity
        self.compression = compression
        self.mode = mode
        self.fuse_op = {
            'add': torch.add,
            'mul': torch.mul,
            'none': lambda x, y: x
        }[self.fuse_mode]

        if self.rank is not None and self.capacity is not None and self.capacity > 0:
            n_coefs = int(self.capacity*coeff_ratio)
            if self.compression == 'vm':
                weights_t = 0.01*torch.randn((n_coefs, self.rank)) # C, R
                matrix_t = 0.01*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in
                if self.fuse_mode == 'mul': # so that it starts with identity
                    matrix_t.fill_(1.0)
                    weights_t.fill_(1.0/self.rank)
                self.register_parameter('weights_t', torch.nn.Parameter(weights_t)) # C, R
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) # R, F_out*F_in
            elif self.compression == 'loe':
                matrix_t = 0.0*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) # R, F_out*F_in
            elif self.compression == 'vm_attention':
                attention_weight = torch.ones((n_coefs, self.rank)) # C, R
                self.register_parameter('attention_weight', torch.nn.Parameter(attention_weight)) # C, R
                weights_t = 0.01*torch.randn((n_coefs, self.rank)) # C, R
                matrix_t = 0.01*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in
                if self.fuse_mode == 'mul': # so that it starts with identity
                    matrix_t.fill_(1.0)
                    weights_t.fill_(1.0/self.rank)
                self.register_parameter('weights_t', torch.nn.Parameter(weights_t)) # C, R
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) # R, F_out*F_in
            elif self.compression == 'vm_noweight':
                matrix_t = 0.000001*torch.randn((self.rank, self.weight.shape[0]*self.weight.shape[1])) # R, F_out*F_in
                self.register_parameter('matrix_t', torch.nn.Parameter(matrix_t)) # R, F_out*F_in
            elif self.compression == 'none':
                self.register_parameter('matrix_t', torch.nn.Parameter(0.0*torch.randn((self.capacity, self.weight.shape[0]*self.weight.shape[1])))) # C, F_out*F_in
            elif self.compression == 'resnet':
                self.register_parameter('resnet_vec', torch.nn.Parameter(0.0*torch.randn((self.capacity, self.weight.shape[0])))) # C, F_out
            elif self.compression == 'cp':
                weights, factors = random_cp((capacity, self.weight.shape[0], self.weight.shape[1]), self.rank, normalise_factors=False) # F_out, F_in
                self.register_parameter(f'lin_w', torch.nn.Parameter(0.01*torch.randn_like(torch.tensor(weights))))
                self.register_parameter(f'lin_f1', torch.nn.Parameter(0.01*torch.randn_like(torch.tensor(factors[0]))))
                self.register_parameter(f'lin_f2', torch.nn.Parameter(0.01*torch.randn_like(torch.tensor(factors[1]))))
                self.register_parameter(f'lin_f3', torch.nn.Parameter(0.01*torch.randn_like(torch.tensor(factors[2]))))
            elif self.compression == 'tucker':
                tmp = tl.decomposition.tucker(self.weight[None].repeat((capacity, 1, 1)), rank=self.rank, init='random', tol=10e-5, random_state=12345, n_iter_max=1)
                self.core = torch.nn.Parameter(0.01*torch.randn_like(tmp.core))
                factors = [0.01*torch.randn_like(_f) for _f in tmp.factors]
                self.factors = torch.nn.ParameterList([torch.nn.Parameter(_f) for _f in factors])
            else:
                raise NotImplementedError

    def _get_delta_weight(self, input_time=None, frame_id=None):
        """Returns the delta weight matrix for a given time index.

        Args:
            input_time: time index of the input tensor. Data range from -1 to 1.
                Tensor of shape (N)
            frame_id: discrete frame index, tensor of shape (N) or scalar
        Returns:
            delta weight matrix of shape (N, F_out, F_in)
        """
        # return self.weight + torch.einsum('tr,rfi->tfi', self.weights_t, self.matrix_t)
        if self.compression == 'vm':
            weights_t = self.weights_t # C, R
            if self.mode == 'interpolation':
                grid_query = input_time.view(1, -1, 1, 1) # 1, N, 1, 1

                weights_t = torch.nn.functional.grid_sample(
                    weights_t.transpose(0, 1).unsqueeze(0).unsqueeze(-1), # 1, R, C, 1
                    torch.cat([torch.zeros_like(grid_query), grid_query], dim=-1),
                    padding_mode='border',
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0).squeeze(-1).transpose(0, 1) # 1, R, N, 1 ->  N, R
            delta_w = self.fuse_op((weights_t @ self.matrix_t).t(), self.weight.view(-1, 1)) # F_out*F_in, C
        elif self.compression == 'loe':
            grid_query = input_time.view(1, -1, 1, 1) # 1, N, 1, 1
            delta_w = torch.nn.functional.grid_sample(
                self.matrix_t.transpose(0, 1).unsqueeze(0).unsqueeze(-1), # 1, F_out*F_in, R, 1
                torch.cat([torch.zeros_like(grid_query), grid_query], dim=-1),
                padding_mode='border',
                mode='nearest',
                align_corners=True
            ).squeeze(0).squeeze(-1) # 1, F_out*F_in, N, 1 ->  F_out*F_in,N
        elif self.compression == 'vm_attention':
            attention = torch.softmax(self.attention_weight @ self.attention_weight.t()/self.rank, dim=0) # C,R @ R,C -> C,C
            weights = attention @ self.weights_t # C,C @ C,R -> C,R
            delta_w = self.fuse_op((weights @ self.matrix_t).t(), self.weight.view(-1, 1))
        elif self.compression == 'vm_noweight':
            delta_w = self.fuse_op(self.matrix_t.t(), self.weight.view(-1, 1)) # F_out*F_in, C
            delta_w = delta_w.sum(1, keepdim=True).repeat(1, self.capacity) # F_out*F_in, C
        elif self.compression == 'none':
            delta_w = self.fuse_op(self.matrix_t.t(), self.weight.view(-1, 1)) # F_out*F_in, C
        elif self.compression == 'cp':
            _weights = getattr(self, f'lin_w')
            _factors = [getattr(self, f'lin_f1'), getattr(self, f'lin_f2'), getattr(self, f'lin_f3')]
            lin_w = tl.cp_to_tensor((_weights, _factors)) # C, F_out, F_in
            delta_w = self.fuse_op(lin_w.view(lin_w.shape[0], -1).t(), self.weight.view(-1, 1)) # F_out*F_in, C
        elif self.compression == 'tucker':
            core = getattr(self, f'core')
            factors = getattr(self, f'factors')
            lin_w = tl.tucker_to_tensor((core, factors)) # C, F_out, F_in
            delta_w = self.fuse_op(lin_w.reshape(lin_w.shape[0], -1).t(), self.weight.view(-1, 1)) # F_out*F_in, C
        else:
            raise NotImplementedError

        mat = delta_w.permute(1, 0).view(-1, *self.weight.shape)
        if mat.shape[0] == 1:
            out = mat[0]
        else:
            if self.mode == 'interpolation':
                out = mat
            else:
                # Clone mat before indexing to avoid inference tensor error
                # when using with torch.inference_mode() or torch.no_grad()
                if not self.training:
                    mat = mat.detach().clone()
                
                # Handle scalar frame_id (int or 0-d tensor)
                if frame_id is not None:
                    if isinstance(frame_id, (int, torch.int64, torch.int32)):
                        frame_idx = int(frame_id)
                    elif isinstance(frame_id, torch.Tensor) and frame_id.dim() == 0:
                        frame_idx = int(frame_id.item())
                    elif isinstance(frame_id, torch.Tensor) and frame_id.dim() == 1:
                        # frame_id is a 1D tensor, use it for indexing
                        out = mat[frame_id]  # (N, F_out, F_in)
                        return out
                    else:
                        frame_idx = frame_id
                    out = mat[frame_idx]  # (F_out, F_in)
                else:
                    # No frame_id provided, return all
                    out = mat

        return out # (F_out, F_in) or (N, F_out, F_in)

    def forward(self, input: torch.Tensor, input_time=None, frame_id=None) -> torch.Tensor:
        """Applies the linear transformation to the incoming data: :math:`y = x(A + \delta A_t)^T + b

        Args:
            input: (B, S, F_in) or (B, F_in)
            input_time: time index of the input tensor. Data range from -1 to 1.
                Tensor of shape (B) or (1)
            frame_id: discrete frame index, tensor of shape (B) or scalar
        Returns:
            output: (B, S, F_out) or (B, F_out)
        """
        if self.rank == 0 or self.capacity == 0 or self.compression == 'resnet':
            return torch.nn.functional.linear(input, self.weight, self.bias)

        weight = self._get_delta_weight(input_time, frame_id) # B, F_out, F_in or (F_out, F_in)
        
        # Ensure weight is 2D for linear operation
        if len(weight.shape) == 2:
            # weight is (F_out, F_in), use directly
            return torch.nn.functional.linear(input, weight, self.bias)
        elif weight.shape[0] == 1:
            # weight is (1, F_out, F_in), squeeze first dimension
            weight_2d = weight.squeeze(0)  # (F_out, F_in)
            if len(weight_2d.shape) != 2:
                # If squeeze resulted in 1D, reshape properly
                weight_2d = weight_2d.view(self.out_features, self.in_features)
            return torch.nn.functional.linear(input, weight_2d, self.bias)
        else:
            # weight is (B, F_out, F_in), handle batched case
            if len(input.shape) == 2:
                # Input is (B, F_in), weight is (B, F_out, F_in)
                # Reshape input to (B, 1, F_in) for batched matmul
                input_3d = input.unsqueeze(1)  # (B, 1, F_in)
                # (B, F_out, F_in) @ (B, F_in, 1) -> (B, F_out, 1)
                output = (weight @ input_3d.permute(0, 2, 1) + self.bias.view(1, -1, 1)).permute(0, 2, 1)  # (B, 1, F_out)
                return output.squeeze(1)  # (B, F_out)
            else:
                # Input is (B, S, F_in), weight is (B, F_out, F_in)
                # (B, F_out, F_in) x (B, F_in, S) -> (B, F_out, S)
                return (weight @ input.permute(0, 2, 1) + self.bias.view(1, -1, 1)).permute(0, 2, 1) # B, S, F_out

    def extra_repr(self) -> str:
        _str = 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        if self.rank is not None and self.capacity is not None:
            _str += ', rank={}, capacity={}, compression={}, mode={}'.format(
                self.rank, self.capacity, self.compression, self.mode
            )
        return _str