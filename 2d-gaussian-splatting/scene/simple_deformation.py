import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------
# Progressive Time Encoding
# -------------------------
class ProgressiveTimePE(nn.Module):
    def __init__(self, freqs, max_steps=5000):
        super().__init__()
        self.register_buffer("freqs", freqs)
        self.max_steps = max_steps
        self.cur_step = 0

    def update_step(self, step):
        self.cur_step = step

    def forward(self, t):
        """
        t: (N, 1)
        """
        device = t.device
        freqs = self.freqs.to(device)

        # progressive mask
        alpha = min(1.0, self.cur_step / self.max_steps)
        k = alpha * len(freqs)
        idx = torch.arange(len(freqs), device=device)
        mask = (1.0 - torch.cos(
            math.pi * (k - idx).clamp(0, 1)
        )) / 2.0

        x = t[..., None] * freqs
        sin = torch.sin(2 * math.pi * x) * mask
        cos = torch.cos(2 * math.pi * x) * mask

        pe = torch.cat([sin, cos], dim=-1).view(t.shape[0], -1)
        return torch.cat([t, pe], dim=-1)

# -------------------------
# Standard PE for position
# -------------------------
def better_poc_fre(x, freqs):
    freqs = freqs.to(x.device)
    x = x[..., None] * freqs
    sin = torch.sin(2 * math.pi * x)
    cos = torch.cos(2 * math.pi * x)
    pe = torch.cat([sin, cos], dim=-1).view(x.shape[0], -1)
    return torch.cat([x[..., 0], pe], dim=-1)

# =========================
# Enhanced SimpleDeformation
# =========================
class SimpleDeformation(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        pos_pe=6,
        time_pe=6,
        max_d_scale=0.5,
        progressive_time=True,
        output_pos=True,
        output_scales=True,
        output_rotations=True,
        output_opacity=True,
        output_shs=True
    ):
        super().__init__()
        self.pos_pe = pos_pe
        self.time_pe = time_pe
        self.max_d_scale = max_d_scale

        # PE frequencies
        self.register_buffer("pos_poc", torch.tensor([2 ** i for i in range(pos_pe)], dtype=torch.float32))
        self.register_buffer("time_poc", torch.tensor([2 ** i for i in range(time_pe)], dtype=torch.float32))

        # Input dim
        pos_dim = 3 * (1 + 2 * pos_pe)
        time_dim = 1 * (1 + 2 * time_pe)
        input_dim = pos_dim + time_dim

        # Progressive time encoding
        self.progressive_time = progressive_time
        if progressive_time:
            self.time_encoder = ProgressiveTimePE(self.time_poc)

        # MLP with skip connection
        self.skips = [D // 2]
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, W))
        for i in range(1, D):
            if i in self.skips:
                self.linears.append(nn.Linear(W + input_dim, W))
            else:
                self.linears.append(nn.Linear(W, W))

        # Output heads
        self.output_pos = output_pos
        self.output_scales = output_scales
        self.output_rotations = output_rotations
        self.output_opacity = output_opacity
        self.output_shs = output_shs

        if output_pos:
            self.pos_head = nn.Linear(W, 3)
        if output_scales:
            self.scale_head = nn.Linear(W, 2)
        if output_rotations:
            self.rot_head = nn.Linear(W, 4)
        if output_opacity:
            self.opacity_head = nn.Linear(W, 1)
        if output_shs:
            self.sh_head = nn.Linear(W, 16*3)

        self._init_outputs()

    def _init_outputs(self):
        for head in getattr(self, '__dict__', {}):
            if isinstance(head, nn.Linear):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        # Or explicitly:
        for head in [getattr(self, h) for h in ['pos_head','scale_head','rot_head','opacity_head','sh_head'] if hasattr(self,h)]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def update(self, step):
        if self.progressive_time:
            self.time_encoder.update_step(step)

    def forward(self, x, t, return_components=False):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x_emb = better_poc_fre(x, self.pos_poc)
        t_emb = self.time_encoder(t) if self.progressive_time else better_poc_fre(t, self.time_poc)
        inputs = torch.cat([x_emb, t_emb], dim=-1)

        h = inputs
        for i, layer in enumerate(self.linears):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([inputs, h], dim=-1)

        outputs = []
        if self.output_pos or return_components:
            outputs.append(self.pos_head(h))
        if self.output_scales or return_components:
            # clamp scale
            outputs.append(torch.tanh(self.scale_head(h)) * math.log(self.max_d_scale))
        if self.output_rotations or return_components:
            outputs.append(self.rot_head(h))
        if self.output_opacity or return_components:
            outputs.append(self.opacity_head(h))
        if self.output_shs or return_components:
            outputs.append(self.sh_head(h).view(-1,16,3))

        return tuple(outputs)
