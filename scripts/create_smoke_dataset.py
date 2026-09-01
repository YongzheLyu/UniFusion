#!/usr/bin/env python3
"""Create an isolated small dataset by linking prepared ExoRecon artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def ensure_link(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise SystemExit(f"existing link points elsewhere: {destination}")
    if destination.exists():
        raise SystemExit(f"refusing to overwrite existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--destination-data-root", type=Path, required=True)
    parser.add_argument("--sequence", default="bike")
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    source = args.source_data_root.expanduser().resolve() / args.sequence / "dataset"
    destination = args.destination_data_root.expanduser().resolve() / args.sequence / "dataset"
    source_frames = source / "frames_output"
    source_images = source / "renamed_images"
    if not source_frames.is_dir() or not source_images.is_dir():
        raise SystemExit(f"prepared source dataset is incomplete: {source}")

    for frame_index in range(args.frames):
        frame = source_frames / f"frame_{frame_index:05d}"
        if not frame.is_dir():
            raise SystemExit(f"missing source frame: {frame}")
        ensure_link(frame, destination / "frames_output" / frame.name)

    camera_dirs = sorted((path for path in source_images.iterdir() if path.is_dir()), key=natural_key)
    if not camera_dirs:
        raise SystemExit(f"no camera directories found in {source_images}")
    for camera in camera_dirs:
        images = sorted(
            (path for path in camera.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
            key=natural_key,
        )
        if len(images) < args.frames:
            raise SystemExit(f"{camera} contains only {len(images)} images")
        for image in images[: args.frames]:
            ensure_link(image, destination / "renamed_images" / camera.name / image.name)

    print(f"linked {args.frames} frames from {len(camera_dirs)} cameras into {destination}")


if __name__ == "__main__":
    main()
