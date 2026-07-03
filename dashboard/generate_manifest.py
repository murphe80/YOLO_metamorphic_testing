#!/usr/bin/env python3
"""
Generate a manifest.json file for the dashboard by scanning the results directory.
This allows the dashboard to dynamically discover available datasets, models, and tests.
"""

import json
from pathlib import Path

SKIP_DIRS = {'config_tuning', 'venn_diagrams', 'visualisation', 'visualization', 'visualizations'}


def scan_model(model_dir: Path) -> set:
    """Scan a model directory for available tests."""
    tests = set()

    for test_dir in model_dir.iterdir():
        if not test_dir.is_dir() or test_dir.name.startswith('.'):
            continue
        if test_dir.name in SKIP_DIRS:
            continue

        stats_file = test_dir / "original_vs_transformed" / "stats.json"
        if stats_file.exists():
            tests.add(test_dir.name)

    return tests


def is_model_results_dir(path: Path) -> bool:
    """Return true for analysis/results/<model> style directories."""
    if not path.is_dir() or path.name.startswith('.') or path.name in SKIP_DIRS:
        return False

    if (path / "original_vs_gt" / "stats.json").exists():
        return True

    return any(
        child.is_dir() and (child / "original_vs_transformed" / "stats.json").exists()
        for child in path.iterdir()
    )


def scan_dataset(dataset_path: Path) -> dict:
    """Scan a dataset directory for available models and tests."""
    models = set()
    tests = set()

    if not dataset_path.exists():
        return {"models": [], "tests": []}

    for model_dir in dataset_path.iterdir():
        if not is_model_results_dir(model_dir):
            continue

        model_tests = scan_model(model_dir)
        models.add(model_dir.name)
        tests.update(model_tests)

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
        Dictionary with top-level models/tests and optional per-dataset entries
    """
    results_path = Path(__file__).parent / results_root
    models = set()
    tests = set()
    datasets = {}

    if results_path.exists():
        if any(is_model_results_dir(child) for child in results_path.iterdir()):
            for model_dir in sorted(results_path.iterdir()):
                if not is_model_results_dir(model_dir):
                    continue

                model_tests = scan_model(model_dir)
                models.add(model_dir.name)
                tests.update(model_tests)
        else:
            for dataset_dir in sorted(results_path.iterdir()):
                if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                    continue
                if dataset_dir.name in SKIP_DIRS:
                    continue

                dataset_info = scan_dataset(dataset_dir)
                if dataset_info["models"]:
                    datasets[dataset_dir.name] = dataset_info
                    models.update(dataset_info["models"])
                    tests.update(dataset_info["tests"])

    manifest = {
        "models": sorted(models),
        "tests": sorted(tests),
    }
    if datasets:
        manifest["datasets"] = datasets

    return manifest


def main():
    """Generate and save the manifest file."""
    manifest = generate_manifest()

    output_path = Path(__file__).parent / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest with {len(manifest['models'])} models, {len(manifest['tests'])} tests")
    if "datasets" in manifest:
        dataset_names = list(manifest["datasets"].keys())
        print(f"Datasets: {', '.join(dataset_names)}")
        for name, info in manifest["datasets"].items():
            print(f"  {name}: {len(info['models'])} models, {len(info['tests'])} tests")


if __name__ == "__main__":
    main()
