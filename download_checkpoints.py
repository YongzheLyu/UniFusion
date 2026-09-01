#!/usr/bin/env python3
"""Download the pretrained checkpoints required by UniFusion."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPTH_ENCODERS = {"small": "vits", "base": "vitb", "large": "vitl", "giant": "vitg"}


def download(url: str, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"[skip] {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[download] {url}\n        -> {destination}", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-encoder", choices=DEPTH_ENCODERS, default="large")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    encoder = DEPTH_ENCODERS[args.depth_encoder]
    depth_name = f"depth_anything_v2_{encoder}.pth"
    downloads = [
        (f"https://huggingface.co/depth-anything/Depth-Anything-V2-{args.depth_encoder.capitalize()}/resolve/main/{depth_name}", ROOT / "Depth-Anything-V2/checkpoints" / depth_name),
        ("https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", ROOT / "mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"),
        ("https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth", ROOT / "mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"),
        ("https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl", ROOT / "mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"),
    ]
    for url, destination in downloads:
        download(url, destination, force=args.force)


if __name__ == "__main__":
    main()

