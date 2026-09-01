#!/usr/bin/env python3
"""Create an isolated experiment workspace backed by read-only cached inputs."""

from __future__ import annotations

import argparse
from pathlib import Path


def link(source: Path, destination: Path) -> None:
    if not source.exists():
        raise SystemExit(f"cached input does not exist: {source}")
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise SystemExit(f"refusing to replace different symlink: {destination}")
    if destination.exists():
        raise SystemExit(f"refusing to replace existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--destination-data-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()

    for sequence in args.sequences:
        source_sequence = args.source_data_root.expanduser().resolve() / sequence
        destination_sequence = args.destination_data_root.expanduser().resolve() / sequence
        source_dataset = source_sequence / "dataset"
        destination_dataset = destination_sequence / "dataset"

        link(source_sequence / "grouped_by_cams", destination_sequence / "grouped_by_cams")
        link(source_dataset / "renamed_images", destination_dataset / "renamed_images")
        link(source_dataset / "frames_output", destination_dataset / "frames_output")
        link(
            source_dataset / f"resfield_rank{args.rank}_priors",
            destination_dataset / f"resfield_rank{args.rank}_priors",
        )
        link(
            source_dataset / "final_dataset" / "mast3r_sfm",
            destination_dataset / "final_dataset" / "mast3r_sfm",
        )
        print(f"linked cached inputs for {sequence} under {destination_sequence}")


if __name__ == "__main__":
    main()
