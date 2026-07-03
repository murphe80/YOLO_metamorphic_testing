#!/usr/bin/env python3
"""
Generate a manifest.json file for the dashboard by scanning the results directory.
This allows the dashboard to dynamically discover available datasets, models, and tests.
"""

import json
from pathlib import Path

SKIP_DIRS = {'config_tuning', 'venn_diagrams'}


def scan_dataset(dataset_path: Path) -> dict:
    """Scan a single dataset directory for available models and tests."""
    models = set()
    tests = set()

    if not dataset_path.exists():
        return {"models": [], "tests": []}

    for model_dir in dataset_path.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        models.add(model_dir.name)

        for test_dir in model_dir.iterdir():
            if not test_dir.is_dir() or test_dir.name.startswith('.'):
                continue

            stats_file = test_dir / "original_vs_transformed" / "stats.json"
            if stats_file.exists():
                tests.add(test_dir.name)

    return {
        "models": sorted(models),
        "tests": sorted(tests),
    }


def generate_manifest(results_root: str = "../analysis/results") -> dict:
    """
    Scan the results directory and generate a manifest of available datasets,
    models, and tests.

    Args:
        results_root: Path to the results root directory (relative to this script)

    Returns:
        Dictionary with 'datasets' mapping dataset names to models/tests
    """
    results_path = Path(__file__).parent / results_root
    datasets = {}

    if results_path.exists():
        for dataset_dir in sorted(results_path.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                continue
            if dataset_dir.name in SKIP_DIRS:
                continue

            dataset_info = scan_dataset(dataset_dir)
            if dataset_info["models"]:
                datasets[dataset_dir.name] = dataset_info

    return {"datasets": datasets}


def main():
    """Generate and save the manifest file."""
    manifest = generate_manifest()

    output_path = Path(__file__).parent / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    dataset_names = list(manifest["datasets"].keys())
    print(f"Generated manifest with {len(dataset_names)} datasets: {', '.join(dataset_names)}")
    for name, info in manifest["datasets"].items():
        print(f"  {name}: {len(info['models'])} models, {len(info['tests'])} tests")


if __name__ == "__main__":
    main()
