#!/usr/bin/env python3
"""Rank frames by horizontal USB scan-line noise and quarantine the worst ones."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FrameScore:
    path: Path
    scanline_ratio: float
    vertical_edge_ratio: float


def score_frame(path: Path) -> FrameScore:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {path}")

    gray_float = gray.astype(np.float32)
    horizontal_edges = np.abs(np.diff(gray_float, axis=1)).mean()
    vertical_edges = np.abs(np.diff(gray_float, axis=0)).mean()
    horizontal_second_diff = np.abs(
        gray_float[:, 2:]
        - (2.0 * gray_float[:, 1:-1])
        + gray_float[:, :-2]
    ).mean()
    vertical_second_diff = np.abs(
        gray_float[2:, :]
        - (2.0 * gray_float[1:-1, :])
        + gray_float[:-2, :]
    ).mean()

    epsilon = np.finfo(np.float32).eps
    return FrameScore(
        path=path,
        scanline_ratio=float(vertical_second_diff / (horizontal_second_diff + epsilon)),
        vertical_edge_ratio=float(vertical_edges / (horizontal_edges + epsilon)),
    )


def image_paths(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def write_manifest(
    manifest: Path, scores: list[FrameScore], rejected_names: set[str]
) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["filename", "scanline_ratio", "vertical_edge_ratio", "decision"]
        )
        for frame in sorted(scores, key=lambda item: item.path.name):
            writer.writerow(
                [
                    frame.path.name,
                    f"{frame.scanline_ratio:.8f}",
                    f"{frame.vertical_edge_ratio:.8f}",
                    "quarantine" if frame.path.name in rejected_names else "keep",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing input images")
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument(
        "--quarantine",
        type=Path,
        help="Destination for rejected images (default: sibling rejected_usb_noise)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV output path (default: sibling usb_noise_filter_manifest.csv)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move rejected images; without this flag, only print a dry-run summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    quarantine = (
        args.quarantine.resolve()
        if args.quarantine
        else source.parent / "rejected_usb_noise"
    )
    manifest = (
        args.manifest.resolve()
        if args.manifest
        else source.parent / "usb_noise_filter_manifest.csv"
    )

    paths = image_paths(source)
    if not 0 < args.target_count <= len(paths):
        raise ValueError(
            f"target-count must be between 1 and {len(paths)}; got {args.target_count}"
        )

    scores = [score_frame(path) for path in paths]
    reject_count = len(scores) - args.target_count
    ranked = sorted(
        scores,
        key=lambda item: (item.scanline_ratio, item.vertical_edge_ratio),
        reverse=True,
    )
    rejected = ranked[:reject_count]
    rejected_names = {frame.path.name for frame in rejected}

    kept_scores = [frame.scanline_ratio for frame in ranked[reject_count:]]
    rejected_scores = [frame.scanline_ratio for frame in rejected]
    print(f"Images found: {len(scores)}")
    print(f"Images to keep: {args.target_count}")
    print(f"Images to quarantine: {reject_count}")
    if rejected_scores:
        print(f"Lowest rejected scan-line ratio: {min(rejected_scores):.6f}")
    if kept_scores:
        print(f"Highest retained scan-line ratio: {max(kept_scores):.6f}")

    if not args.apply:
        print("Dry run only; pass --apply to write the manifest and move images.")
        return

    quarantine.mkdir(parents=True, exist_ok=True)
    collisions = [frame.path.name for frame in rejected if (quarantine / frame.path.name).exists()]
    if collisions:
        raise FileExistsError(
            f"Quarantine already contains {len(collisions)} selected image(s); refusing to overwrite"
        )

    write_manifest(manifest, scores, rejected_names)
    for frame in rejected:
        shutil.move(str(frame.path), str(quarantine / frame.path.name))

    remaining = image_paths(source)
    moved = image_paths(quarantine)
    if len(remaining) != args.target_count or len(moved) < reject_count:
        raise RuntimeError(
            f"Post-move validation failed: {len(remaining)} remain and {len(moved)} are quarantined"
        )

    print(f"Manifest: {manifest}")
    print(f"Quarantine: {quarantine}")


if __name__ == "__main__":
    main()
