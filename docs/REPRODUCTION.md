# Reproducing UniFusion experiments

## Stages

1. `preprocess` builds the reusable temporal pickle from per-frame MASt3R data.
2. `align` optimizes time-conditioned charts and writes rank-specific priors.
3. `organize` converts alignment outputs to the layout consumed by 2DGS.
4. `train` refines and densifies the 2D Gaussian representation for 15k steps.
5. `render` exports held-out RGB and depth predictions.
6. `evaluate` computes the main-table full-frame RGB metrics. Pass
   `--evaluate-depth --semidense-root ...` to additionally evaluate depth.
7. `summarize` aggregates all requested sequences into one JSON file.

Use `--resume` after preemption. A stage marker is written only after its command
returns successfully. Remove only that marker if a completed artifact needs to be
regenerated.

## Main setting

The paper setting is fully described by `configs/paper/exorecon.yaml`: rank 4,
learned eight-dimensional time encoding, hinge depth-order supervision, and SSI
depth regularization with weight 2.0.
Following the experiment sheet, cooking, cpr, and soccer use 10k 2DGS iterations;
dance, piano, and bike retain the 15k setting. Use the config unmodified for the
main table.

```bash
python scripts/reproduce_exorecon.py \
  --config configs/paper/exorecon.yaml \
  --data-root "$DATA_ROOT" \
  --resume
```

To test another rank, copy the paper config to a new file and change both
`temporal_alignment.rank` and `refinement.experiment_name`. This preserves the
main configuration as an auditable artifact.

## SLURM

The supplied array job runs one sequence per GPU. Cluster-specific partition,
account, and QoS values are deliberately omitted; provide them through `sbatch`
or a local wrapper.

```bash
DATA_ROOT=/datasets/ExoReconstruction \
CONDA_ENV=unifusion \
sbatch --partition=<partition> --array=0-5 cluster/reproduce_exorecon.slurm
```

## Reproducibility record

Each invocation writes a JSON manifest under `results/manifests/` containing the
resolved config, exact commands, Python/platform details, selected stages, and
`CUDA_VISIBLE_DEVICES`. Archive this manifest with the checkpoint and metric
JSON. Also record `nvidia-smi`, the Git commit, and dataset checksums for a paper
artifact submission.

## Expected metrics

`configs/paper/exorecon_rank4_reference.json` records the user-provided
`plus ssi loss` main-experiment row. Its six-sequence mean is PSNR 31.9017,
SSIM 0.95517, and LPIPS 0.0642. Other ablations and EgoHuman experiments are
intentionally outside the current reproduction scope. Never claim reproduction
based only on successful execution.

## Common failures

- `CUDA_HOME` or `nvcc` missing: install the CUDA 11.8 toolkit and rerun
  `install.py`.
- Native extension import error: rebuild after activating the exact environment;
  do not reuse build artifacts from another PyTorch/CUDA pair.
- Out of memory during alignment: use the documented memory-efficient temporal
  mode or reduce the preprocessing batch size, but report the deviation.
- Missing `charts_data.npz`: rerun `organize` after checking that `align` finished.
- Empty evaluation set: verify the frame range and train stride in the paper
  config, especially the shorter soccer sequence.
