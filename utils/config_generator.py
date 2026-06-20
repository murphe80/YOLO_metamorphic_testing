#!/usr/bin/env python3
"""
Configuration Generator for Metamorphic Test Tuning

Generates 10 configuration variants per transformation with linear parameter sweeps.
For multi-parameter transformations, varies one parameter at a time to maintain 10 total runs.

Usage:
    from utils.config_generator import generate_all_configs, generate_configs

    # Generate configs for all transformations
    all_configs = generate_all_configs()

    # Generate configs for specific transformation
    resolution_configs = generate_configs("imgResolution")
"""

from typing import Dict, List
import numpy as np


def generate_configs(transformation_name: str) -> List[Dict]:
    """
    Generate 10 configuration variants for a specific transformation.

    Args:
        transformation_name: Name of the transformation (e.g., "imgResolution")

    Returns:
        List of 10 config dictionaries with variant_id included

    Raises:
        ValueError: If transformation_name is not recognized
    """
    generators = {
        "imgGrayscale": _generate_grayscale_configs,
        "imgColorWheel": _generate_colorwheel_configs,
        "imgResolution": _generate_resolution_configs,
        "imgDarkBright": _generate_darkbright_configs,
        "imgMotionBlur": _generate_motionblur_configs,
        "imgStretch": _generate_stretch_configs,
    }

    if transformation_name not in generators:
        raise ValueError(
            f"Unknown transformation: {transformation_name}. "
            f"Available: {list(generators.keys())}"
        )

    configs = generators[transformation_name]()

    # Add variant_id to each config
    for i, config in enumerate(configs):
        config["variant_id"] = f"{transformation_name}_{i}"

    return configs


def generate_all_configs() -> Dict[str, List[Dict]]:
    """
    Generate configuration variants for all supported transformations.

    Returns:
        Dictionary mapping transformation names to lists of config variants
        {
            "imgResolution": [{"size_ratio": 0.1, "variant_id": "..."}, ...],
            "imgColorWheel": [...],
            ...
        }
    """
    transformations = [
        "imgGrayscale",
        "imgColorWheel",
        "imgResolution",
        "imgDarkBright",
        "imgMotionBlur",
        "imgStretch",
    ]

    all_configs = {}
    for transform in transformations:
        all_configs[transform] = generate_configs(transform)

    return all_configs


def _generate_grayscale_configs() -> List[Dict]:
    """
    Grayscale has no parameters - return single identity config.
    Replicate 10 times to maintain uniform structure.
    """
    return [{} for _ in range(10)]


def _generate_colorwheel_configs() -> List[Dict]:
    """
    imgColorWheel: Vary hue_rotation from 0 to 180 degrees (OpenCV hue range).
    Linear sweep: [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    """
    hue_rotations = np.linspace(0, 180, 10, dtype=int)
    return [{"hue_rotation": int(hue)} for hue in hue_rotations]


def _generate_resolution_configs() -> List[Dict]:
    """
    imgResolution: Vary size_ratio from 0.1 to 1.0.
    Linear sweep: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    """
    size_ratios = np.linspace(0.1, 1.0, 10)
    return [{"size_ratio": float(ratio)} for ratio in size_ratios]


def _generate_darkbright_configs() -> List[Dict]:
    """
    imgDarkBright: Vary gamma from 0.5 to 5.0 (fix method='gamma').
    Linear sweep: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    Multi-parameter transformation (brightness_constant, brightness_multiplier, gamma, method).
    Decision: Fix method='gamma', vary only gamma parameter.
    """
    gammas = np.linspace(0.5, 5.0, 10)
    return [
        {
            "gamma": float(gamma),
            "method": "gamma",
            "brightness_constant": 0,
            "brightness_multiplier": 1,
        }
        for gamma in gammas
    ]


def _generate_motionblur_configs() -> List[Dict]:
    """
    imgMotionBlur: Vary kernel_size from 3 to 21 (fix orientation=0).
    Linear sweep: [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] (odd numbers only).

    Multi-parameter transformation (kernel_size, orientation).
    Decision: Fix orientation=0 (horizontal blur), vary kernel_size.
    """
    kernel_sizes = np.linspace(3, 21, 10, dtype=int)
    # Ensure odd numbers (required by the transformation)
    kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]

    return [{"kernel_size": int(k), "orientation": 0} for k in kernel_sizes]


def _generate_stretch_configs() -> List[Dict]:
    """
    imgStretch: Vary width_ratio from 1.0 to 2.8 (fix height_ratio=1.0).
    Linear sweep: [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]

    Multi-parameter transformation (height_ratio, width_ratio).
    Decision: Fix height_ratio=1.0 (no vertical stretch), vary width_ratio.
    """
    width_ratios = np.linspace(1.0, 2.8, 10)
    return [
        {"height_ratio": 1.0, "width_ratio": float(ratio)}
        for ratio in width_ratios
    ]


def get_parameter_name(transformation_name: str) -> str:
    """
    Get the primary parameter name being varied for a transformation.
    Useful for plotting and analysis.

    Args:
        transformation_name: Name of the transformation

    Returns:
        Parameter name (e.g., "hue_rotation", "size_ratio", etc.)
    """
    param_map = {
        "imgGrayscale": "none",
        "imgColorWheel": "hue_rotation",
        "imgResolution": "size_ratio",
        "imgDarkBright": "gamma",
        "imgMotionBlur": "kernel_size",
        "imgStretch": "width_ratio",
    }
    return param_map.get(transformation_name, "unknown")


if __name__ == "__main__":
    # Test configuration generation
    print("=" * 80)
    print("Configuration Generator Test")
    print("=" * 80)

    all_configs = generate_all_configs()

    total_configs = 0
    for transform_name, configs in all_configs.items():
        print(f"\n{transform_name}:")
        print(f"  Generated {len(configs)} configurations")
        print(f"  Primary parameter: {get_parameter_name(transform_name)}")
        print(f"  Sample configs:")

        # Show first, middle, and last config
        if configs:
            for idx in [0, len(configs) // 2, len(configs) - 1]:
                config = configs[idx]
                variant_id = config.pop("variant_id")
                print(f"    [{idx}] {variant_id}: {config}")
                config["variant_id"] = variant_id  # Restore

        total_configs += len(configs)

    print(f"\n{'=' * 80}")
    print(f"Total configurations generated: {total_configs}")
    print(f"Expected: {6 * 10} (6 transformations × 10 variants)")
    print("=" * 80)
