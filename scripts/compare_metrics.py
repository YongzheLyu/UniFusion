#!/usr/bin/env python3
"""Compare a reproduced UniFusion summary with the frozen reference metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOLERANCES = {
    "psnr": 0.20,
    "ssim": 0.005,
    "lpips": 0.010,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path, nargs="?", default=ROOT / "results/exorecon_rank4_summary.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "configs/paper/exorecon_rank4_reference.json")
    args = parser.parse_args()
    actual = json.loads(args.actual.read_text())
    reference = json.loads(args.reference.read_text())
    actual_rows = {row.get("sequence", row.get("seq")): row for row in actual["rows"]}
    reference_rows = {row["sequence"]: row for row in reference["rows"]}
    failures: list[str] = []
    print(f"{'sequence':<10} {'metric':<22} {'actual':>11} {'reference':>11} {'delta':>11} {'tol':>8}")
    for sequence, expected in reference_rows.items():
        if sequence not in actual_rows:
            failures.append(f"missing sequence: {sequence}")
            continue
        observed = actual_rows[sequence]
        for metric, tolerance in DEFAULT_TOLERANCES.items():
            delta = float(observed[metric]) - float(expected[metric])
            print(f"{sequence:<10} {metric:<22} {float(observed[metric]):11.5f} {float(expected[metric]):11.5f} {delta:11.5f} {tolerance:8.3f}")
            if abs(delta) > tolerance:
                failures.append(f"{sequence}/{metric}: |{delta:.6g}| > {tolerance}")
    if failures:
        print("\nRegression check failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("\nAll reproduced metrics are within the configured regression tolerances.")


if __name__ == "__main__":
    main()
