#!/usr/bin/env python3
"""
Generate a manifest.json file for the dashboard by scanning the results directory.
This allows the dashboard to dynamically discover available models and tests.
"""

import json
import os
from pathlib import Path


def generate_manifest(results_dir: str = "../analysis/results") -> dict:
    """
    Scan the results directory and generate a manifest of available models and tests.

    Args:
        results_dir: Path to the results directory (relative to this script)

    Returns:
        Dictionary with 'models' and 'tests' lists
    """
    results_path = Path(__file__).parent / results_dir

    models = set()
    tests = set()

    # Scan directory structure
    if results_path.exists():
        for model_dir in results_path.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue

            # Skip non-model directories
            if model_dir.name in ['config_tuning', 'venn_diagrams']:
                continue

            models.add(model_dir.name)

            # Scan for tests within this model
            for test_dir in model_dir.iterdir():
                if not test_dir.is_dir() or test_dir.name.startswith('.'):
                    continue

                # Check if this test has valid data
                stats_file = test_dir / "original_vs_transformed" / "stats.json"
                if stats_file.exists():
                    tests.add(test_dir.name)

    manifest = {
        "models": sorted(list(models)),
        "tests": sorted(list(tests))
    }

    return manifest


def main():
    """Generate and save the manifest file."""
    manifest = generate_manifest()

    output_path = Path(__file__).parent / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest with {len(manifest['models'])} models and {len(manifest['tests'])} tests")
    print(f"Models: {', '.join(manifest['models'])}")
    print(f"Tests: {', '.join(manifest['tests'])}")


if __name__ == "__main__":
    main()
