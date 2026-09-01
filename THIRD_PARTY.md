# Third-party components

UniFusion contains vendored and adapted third-party research code. The license
closest to each component takes precedence over the root license.

| Component | Location | Upstream | License file |
|---|---|---|---|
| MAtCha-derived chart pipeline | `matcha/`, `scripts/` | `anttwo/MAtCha` | root `LICENSE` and upstream notice |
| MASt3R / DUSt3R | `mast3r/` | `naver/mast3r` | `mast3r/LICENSE`, nested licenses |
| Depth Anything V2 | `Depth-Anything-V2/` | `DepthAnything/Depth-Anything-V2` | `Depth-Anything-V2/LICENSE` |
| 2D Gaussian Splatting | `2d-gaussian-splatting/` | `hbb1/2d-gaussian-splatting` | `2d-gaussian-splatting/LICENSE.md` |
| CUDA extensions | `2d-gaussian-splatting/submodules/` | upstream subprojects | component license/readme |

Several dependencies use research or non-commercial licenses. Do not describe
the entire repository as MIT-only and do not remove upstream notices.

Before public release, record the exact upstream revisions, verify permission to
redistribute modifications, and add notices required by checkpoints and assets.

