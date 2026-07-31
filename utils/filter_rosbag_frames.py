#!/usr/bin/env python3
"""Select an evenly spaced subset of extracted ROS bag image frames."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT / "data/base_images/comp_data/second_trackd_0_frames"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/comp_data"
FRAME_PATTERN = re.compile(r"^frame_(\d+)(?:_.+)?\.[^.]+$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def index_frames(source_dir: Path) -> dict[int, Path]:
    """Map the numeric frame prefix in each filename to its source path."""
    frames: dict[int, Path] = {}
    duplicates: list[int] = []

    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        match = FRAME_PATTERN.match(path.name)
        if match is None:
            continue

        frame_number = int(match.group(1))
        if frame_number in frames:
            duplicates.append(frame_number)
        else:
            frames[frame_number] = path

    if duplicates:
        duplicate_text = ", ".join(str(number) for number in duplicates[:10])
        raise ValueError(
            f"Multiple files represent the same frame number: {duplicate_text}"
        )
    if not frames:
        raise ValueError(f"No frame_NNNNNN image files found in {source_dir}")
    return frames


def evenly_spaced_indices(start: int, end: int, count: int) -> list[int]:
    """Return exactly count integer indices spanning start through end.

    When an exact fixed step is impossible, nearest-integer positions distribute
    the rounding error across the range. For example, a nominal step of 4.51
    produces gaps of four and five frames instead of clustering the selection.
    """
    if start < 0:
        raise ValueError("Start frame cannot be negative.")
    if end < start:
        raise ValueError("End frame must be greater than or equal to start frame.")

    available = end - start + 1
    if count > available:
        raise ValueError(
            f"Cannot select {count} unique frames from a range containing "
            f"only {available} frames."
        )
    if count == 1:
        return [start]

    span = end - start
    intervals = count - 1
    # Integer half-up rounding avoids floating-point drift and includes both ends.
    return [
        start + (position * span + intervals // 2) // intervals
        for position in range(count)
    ]


def validate_selection(
    requested_indices: list[int], frames: dict[int, Path]
) -> list[Path]:
    missing = [index for index in requested_indices if index not in frames]
    if missing:
        preview = ", ".join(str(index) for index in missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise ValueError(
            f"{len(missing)} selected frame(s) are missing from the source: "
            f"{preview}{suffix}"
        )
    return [frames[index] for index in requested_indices]


def describe_selection(indices: list[int]) -> str:
    if len(indices) == 1:
        return "one selected frame"
    gaps = Counter(second - first for first, second in zip(indices, indices[1:]))
    gap_text = ", ".join(
        f"{gap}-frame gap x {frequency}" for gap, frequency in sorted(gaps.items())
    )
    return f"spacing: {gap_text}"


def copy_selection(
    selected_paths: list[Path], output_dir: Path, overwrite: bool
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")

    existing_files = list(output_dir.iterdir()) if output_dir.exists() else []
    if existing_files and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Use --overwrite to replace files with matching names."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for source_path in selected_paths:
        shutil.copy2(source_path, output_dir / source_path.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an evenly spaced subset of extracted frame_NNNNNN images. "
            "The start and end frames are both included when count is greater "
            "than one."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"source frame directory (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--start-frame", type=int, default=1197, help="first frame (default: 1197)"
    )
    parser.add_argument(
        "--end-frame", type=int, default=2830, help="last frame (default: 2830)"
    )
    parser.add_argument(
        "--count",
        type=positive_int,
        default=363,
        help="number of frames to select (default: 363)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of files with matching names",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the selection without copying files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        indices = evenly_spaced_indices(
            args.start_frame, args.end_frame, args.count
        )
        frames = index_frames(source_dir)
        selected_paths = validate_selection(indices, frames)

        print(
            f"Selected {len(selected_paths)} frames from {indices[0]} through "
            f"{indices[-1]} ({describe_selection(indices)})"
        )
        print(f"Source: {source_dir}")
        print(f"Output: {output_dir}")

        if args.dry_run:
            print("Dry run: no files copied.")
            return 0

        copy_selection(selected_paths, output_dir, args.overwrite)
        print(f"Done: copied {len(selected_paths)} images to {output_dir}")
        return 0
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
