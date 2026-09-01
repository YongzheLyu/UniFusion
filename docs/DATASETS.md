# Dataset contract

UniFusion does not redistribute ExoReconstruction images, calibration, dynamic
masks, or semidense ground truth. Obtain them under their original terms.

Each sequence is rooted at `<DATA_ROOT>/<sequence>/dataset`. The temporal
preprocessor consumes numbered `frame_XXXXX` directories under `frames_output`.
Every frame must contain a `mast3r_sfm` scene with images, camera metadata,
pointmaps, and sparse reconstruction files expected by `matcha`.

The six paper sequence aliases and sparse-depth IDs are fixed in
`configs/paper/exorecon.yaml`. Do not silently change the frame ranges when
reporting paper numbers.

Validate the tree without running a model:

```bash
python scripts/reproduce_exorecon.py \
  --data-root "$DATA_ROOT" \
  --semidense-root "$SEMIDENSE_ROOT" \
  --stages preprocess \
  --dry-run
```

Generated temporal priors may be hundreds of gigabytes and should remain next to
the dataset. They are never expected inside the Git checkout.

Human masks are optional for training. When foreground-only evaluation is used,
pass an explicit `--mask_dir` to `scripts/evaluate_rendering.py`; no private
machine default is assumed.

