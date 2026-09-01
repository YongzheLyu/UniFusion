# UniFusion

Official research implementation for dynamic scene reconstruction with temporally
conditioned 2D Gaussian surfels. UniFusion combines multi-view chart priors,
low-rank temporal deformation fields, depth-order supervision, and 2D Gaussian
refinement to recover time-varying appearance and geometry.

> **Release checklist:** add the final paper title, author list, project URL,
> citation, checkpoint URLs, and reference metric table before making the
> repository public. The code and experiment recipes are structured for internal
> reproduction now.

## Highlights

- Temporal preprocessing and chart alignment for multi-camera sequences.
- Low-rank, time-conditioned deformation with optional depth-order occlusion loss.
- 2D Gaussian refinement, novel-view rendering, and dynamic depth export.
- One reproducible command for the six ExoReconstruction sequences.
- Main-table RGB evaluation (PSNR, SSIM, LPIPS), with optional sparse-depth evaluation.
- Parameterized SLURM jobs with no user- or cluster-specific absolute paths.

Datasets, pretrained weights, generated priors, checkpoints, renders, and logs
are deliberately excluded from Git.

## Installation

The reference environment is Linux x86-64, Python 3.9, PyTorch 2.0.1, and CUDA
11.8. Building the custom extensions requires the CUDA toolkit including `nvcc`;
the CUDA runtime package alone is insufficient.

```bash
git clone <PUBLIC_REPOSITORY_URL> UniFusion
cd UniFusion
conda env create -f environment.yml
conda activate unifusion
python install.py --env-name unifusion
python download_checkpoints.py
python scripts/check_environment.py --strict-gpu
```

If the Conda environment already exists, skip `conda env create`. Installation
stops on the first failed dependency build instead of silently continuing.

## Dataset layout

```text
<DATA_ROOT>/
  bike/
    grouped_by_cams/
      <camera-0>/*.jpg
      <camera-1>/*.jpg
      <camera-2>/*.jpg
      <camera-3>/*.jpg
    dataset/
      renamed_images/
      frames_output/
        frame_00000/mast3r_sfm/...
        frame_00001/mast3r_sfm/...
        preprocessed_temporal_data.pkl
      final_dataset/mast3r_sfm/...
  cooking/{grouped_by_cams,dataset}/...
  cpr/{grouped_by_cams,dataset}/...
  dance/{grouped_by_cams,dataset}/...
  piano/{grouped_by_cams,dataset}/...
  soccer/{grouped_by_cams,dataset}/...

<SEMIDENSE_ROOT>/
  cmu_bike14_2/semidense_points.csv.gz
  iiith_cooking_123_4/semidense_points.csv.gz
  nus_cpr_08_1/semidense_points.csv.gz
  uniandes_dance_020_18/semidense_points.csv.gz
  indiana_music_14_3/semidense_points.csv.gz
  cmu_soccer07_3/semidense_points.csv.gz
```

See [docs/DATASETS.md](docs/DATASETS.md) for the exact file contract.

## Reproduce the paper pipeline

Validate inputs and print every command without starting GPU work:

```bash
python scripts/reproduce_exorecon.py \
  --config configs/paper/exorecon.yaml \
  --data-root /path/to/ExoReconstruction \
  --dry-run
```

Run one sequence end to end:

```bash
python scripts/reproduce_exorecon.py \
  --config configs/paper/exorecon.yaml \
  --data-root /path/to/ExoReconstruction \
  --sequences bike \
  --stages prepare preprocess align organize finalize train render evaluate summarize \
  --resume
```

Run all six sequences as a SLURM array:

```bash
DATA_ROOT=/path/to/ExoReconstruction \
CONDA_ENV=unifusion \
sbatch --array=0-5 cluster/reproduce_exorecon.slurm
```

For rank 4, results are written to:

```text
<DATA_ROOT>/<sequence>/dataset/
  resfield_rank4_priors/
  final_dataset/free_gaussians_resfield_rank4/
    test/ours_<iteration>/
      renders/
      gt/
      depth/
      eval_rendering.json
      sparse_depth_eval/depth_eval_results.json  # only with --evaluate-depth
```

Cooking, CPR, and soccer use 10k iterations; dance, piano, and bike use 15k, as
recorded in the main-experiment sheet. The target is the `plus ssi loss` row:
PSNR 31.9017, SSIM 0.95517, and LPIPS 0.0642 on average. The aggregate table is saved as
`results/exorecon_rank4_summary.json`. See
[docs/REPRODUCTION.md](docs/REPRODUCTION.md) for stages, ablations, resuming,
expected artifacts, and troubleshooting.

Compare a completed run with the frozen pre-release reference:

```bash
python scripts/compare_metrics.py results/exorecon_rank4_summary.json
```

Before a full run, submit the isolated three-frame smoke test. It links existing
prepared inputs read-only and writes everything below `work/smoke_data`:

```bash
sbatch cluster/smoke_bike.slurm
```

The smoke recipe uses 3 chart-alignment iterations, the standard 3,000-iteration
coarse initialization, and 20 fine-stage iterations. It checks execution only
and must never be reported as a paper result. Individual stages can be rerun,
for example `STAGES='render evaluate' sbatch cluster/smoke_bike.slurm`.

## Static or sparse-view pipeline

The original MAtCha-compatible entry point remains available:

```bash
python train.py -s /path/to/images -o outputs/example
```

For temporal experiments, use `scripts/reproduce_exorecon.py`; it validates the
input tree, records commands, propagates failures, and resumes from artifacts.

## Repository layout

```text
2d-gaussian-splatting/  2D Gaussian training, rendering, and meshing
matcha/                  chart representation and temporal deformation modules
mast3r/                  vendored MASt3R-SfM dependency
Depth-Anything-V2/       vendored monocular-depth dependency
configs/                 algorithm and immutable paper configurations
scripts/                 preprocessing, orchestration, and evaluation
cluster/                 scheduler templates
tools/                   optional conversion and visualization utilities
docs/                    dataset, reproduction, and development documentation
tests/                   CPU-only release smoke tests
```

The upstream directory names are retained because their CUDA build scripts and
imports depend on those paths. Project-specific orchestration lives in `scripts/`
and core temporal changes live in `matcha/` and `2d-gaussian-splatting/`.

## Reproducibility policy

- Paper configurations under `configs/paper/` are never modified by a run.
- Every run writes commands and environment metadata beside its results.
- `--resume` skips only stages with their expected completion artifact.
- `--dry-run` validates paths and prints commands without GPU execution.
- Evaluation does not train or overwrite checkpoints.

## License and acknowledgement

This repository contains code derived from MAtCha, MASt3R/DUSt3R, Depth Anything
V2, 2D Gaussian Splatting, and Tetra-NeRF. Their licenses are not uniform and
some restrict commercial use. Read [THIRD_PARTY.md](THIRD_PARTY.md) and the
license file in each vendored directory. The root [LICENSE](LICENSE) applies only
where a component-specific license does not supersede it.

Please cite the UniFusion paper after its final citation is added, together with
the upstream projects used by your experiment.
