# Contributing

Keep generated data outside the repository. Before opening a pull request, run:

```bash
python scripts/check_environment.py
python -m compileall -q train.py scripts matcha 2d-gaussian-splatting
python -m unittest discover -s tests -v
```

Changes affecting paper numbers must include the exact command, configuration,
seed, GPU model, CUDA/PyTorch versions, and before/after metrics. Never commit
datasets, weights, renders, logs, credentials, or machine-specific paths.

