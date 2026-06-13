#!/usr/bin/env python3
"""
Model registry for YOLO metamorphic testing framework.
Provides centralized access to model configurations.

Usage:
    from utils.model_registry import get_models, get_model_by_name

    # Get all enabled models
    models = get_models()

    # Get specific model
    model = get_model_by_name("yolov5")
    print(model["weights"])  # models/yolov5/yolov5_param_kfold.pt

    # Get model names only
    names = get_model_names()  # ['yolov5', 'yolov8', 'yolov12']
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


_CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.json"
_CACHE = None


def _load_config() -> Dict:
    """Load model configuration from JSON file (cached)."""
    global _CACHE
    if _CACHE is None:
        with open(_CONFIG_PATH) as f:
            _CACHE = json.load(f)
    return _CACHE


def get_models(enabled_only: bool = True) -> List[Dict]:
    """
    Get list of all model configurations.

    Args:
        enabled_only: If True, only return models where enabled=True

    Returns:
        List of model configuration dictionaries
    """
    config = _load_config()
    models = config["models"]

    if enabled_only:
        models = [m for m in models if m.get("enabled", True)]

    return models


def get_model_by_name(name: str) -> Optional[Dict]:
    """
    Get model configuration by name.

    Args:
        name: Model name (e.g., 'yolov5', 'yolov8')

    Returns:
        Model configuration dict or None if not found
    """
    models = get_models(enabled_only=False)
    for model in models:
        if model["name"] == name:
            return model
    return None


def get_model_by_version(version: str) -> Optional[Dict]:
    """
    Get model configuration by version.

    Args:
        version: Model version (e.g., 'v5', 'v8')

    Returns:
        Model configuration dict or None if not found
    """
    models = get_models(enabled_only=False)
    for model in models:
        if model["version"] == version:
            return model
    return None


def get_model_names(enabled_only: bool = True) -> List[str]:
    """
    Get list of model names.

    Args:
        enabled_only: If True, only return enabled models

    Returns:
        List of model names (e.g., ['yolov5', 'yolov8', 'yolov12'])
    """
    models = get_models(enabled_only=enabled_only)
    return [m["name"] for m in models]


def get_model_versions(enabled_only: bool = True) -> List[str]:
    """
    Get list of model versions.

    Args:
        enabled_only: If True, only return enabled models

    Returns:
        List of model versions (e.g., ['v5', 'v8', 'v12'])
    """
    models = get_models(enabled_only=enabled_only)
    return [m["version"] for m in models]


def get_default_inference_params() -> Dict:
    """
    Get default inference parameters.

    Returns:
        Dictionary of default parameters (conf_thres, iou_thres, device)
    """
    config = _load_config()
    return config.get("default_inference_params", {})


def validate_model_files(model_name: str) -> Dict[str, bool]:
    """
    Check if model files exist on disk.

    Args:
        model_name: Model name to validate

    Returns:
        Dictionary with 'weights' and 'script' keys indicating existence
    """
    model = get_model_by_name(model_name)
    if not model:
        return {"exists": False, "weights": False, "script": False}

    root = Path(__file__).parent.parent
    weights_path = root / model["weights"]
    script_path = root / model["inference_script"]

    return {
        "exists": True,
        "weights": weights_path.exists(),
        "script": script_path.exists(),
        "weights_path": str(weights_path),
        "script_path": str(script_path)
    }


if __name__ == "__main__":
    # CLI tool for querying model registry
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Query model registry")
    parser.add_argument("--list", action="store_true", help="List all model names")
    parser.add_argument("--list-versions", action="store_true", help="List all model versions")
    parser.add_argument("--get", type=str, help="Get model config by name")
    parser.add_argument("--validate", type=str, help="Validate model files exist")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--include-disabled", action="store_true", help="Include disabled models")

    args = parser.parse_args()

    enabled_only = not args.include_disabled

    if args.list:
        names = get_model_names(enabled_only=enabled_only)
        if args.json:
            print(json.dumps(names, indent=2))
        else:
            print("\n".join(names))

    elif args.list_versions:
        versions = get_model_versions(enabled_only=enabled_only)
        if args.json:
            print(json.dumps(versions, indent=2))
        else:
            print("\n".join(versions))

    elif args.get:
        model = get_model_by_name(args.get)
        if model:
            print(json.dumps(model, indent=2))
        else:
            print(f"Model '{args.get}' not found", file=sys.stderr)
            sys.exit(1)

    elif args.validate:
        result = validate_model_files(args.validate)
        print(json.dumps(result, indent=2))
        if not result.get("exists") or not (result.get("weights") and result.get("script")):
            sys.exit(1)

    else:
        # Default: show all models
        models = get_models(enabled_only=enabled_only)
        print(json.dumps(models, indent=2))
