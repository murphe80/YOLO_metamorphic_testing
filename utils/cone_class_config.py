"""Validated semantic cone-class configuration shared by colour utilities."""

import json
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "cone_classes.json"


def load_cone_class_config(config_path, dataset):
    """Load one dataset's local label IDs and semantic transformations."""
    path = Path(config_path)
    with path.open() as handle:
        config = json.load(handle)

    classes = config.get("classes")
    aliases = config.get("aliases", {})
    datasets = config.get("datasets")
    transformations = config.get("transformations")
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"{path}: 'classes' must be a non-empty object")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"{path}: 'datasets' must be a non-empty object")
    if not isinstance(transformations, dict) or not transformations:
        raise ValueError(f"{path}: 'transformations' must be a non-empty object")
    if dataset not in datasets:
        raise ValueError(
            f"{path}: no cone-class schema for dataset {dataset!r}; "
            "add it under 'datasets' before generating transformations"
        )

    canonical_names = set(classes)
    for alias, canonical in aliases.items():
        if canonical not in canonical_names:
            raise ValueError(
                f"{path}: alias {alias!r} refers to unknown class {canonical!r}"
            )

    ordered_names = datasets[dataset]
    if not isinstance(ordered_names, list) or not ordered_names:
        raise ValueError(f"{path}: dataset {dataset!r} must define an ordered class list")

    normalized_names = [aliases.get(name, name) for name in ordered_names]
    unknown = sorted(set(normalized_names) - canonical_names)
    if unknown:
        raise ValueError(
            f"{path}: dataset {dataset!r} uses unknown classes: {unknown}"
        )
    duplicates = sorted(
        name for name in set(normalized_names) if normalized_names.count(name) > 1
    )
    if duplicates:
        raise ValueError(
            f"{path}: dataset {dataset!r} repeats classes: {duplicates}"
        )

    class_id_to_name = dict(enumerate(normalized_names))
    class_name_to_id = {name: class_id for class_id, name in class_id_to_name.items()}

    normalized_transformations = {}
    for name, definition in transformations.items():
        pairs = definition.get("pairs") if isinstance(definition, dict) else None
        if pairs is None and isinstance(definition, dict):
            pairs = [
                {
                    "source": definition.get("source"),
                    "target": definition.get("target"),
                }
            ]
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"{path}: transformation {name!r} has no class pairs")

        mapping = {}
        for pair in pairs:
            source = aliases.get(pair.get("source"), pair.get("source"))
            target = aliases.get(pair.get("target"), pair.get("target"))
            if source not in canonical_names or target not in canonical_names:
                raise ValueError(
                    f"{path}: transformation {name!r} uses an unknown class"
                )
            if source in mapping:
                raise ValueError(
                    f"{path}: transformation {name!r} repeats source {source!r}"
                )
            if target not in class_name_to_id:
                raise ValueError(
                    f"{path}: dataset {dataset!r} has no target class {target!r} "
                    f"required by transformation {name!r}"
                )
            mapping[source] = target
        normalized_transformations[name] = mapping

    hues = {}
    for name, definition in classes.items():
        hue = definition.get("hue") if isinstance(definition, dict) else None
        if hue is not None and (
            not isinstance(hue, int) or isinstance(hue, bool) or not 0 <= hue <= 179
        ):
            raise ValueError(
                f"{path}: class {name!r} hue must be an integer from 0 to 179"
            )
        hues[name] = hue

    for transformation, mapping in normalized_transformations.items():
        for target in mapping.values():
            if hues[target] is None:
                raise ValueError(
                    f"{path}: transformation {transformation!r} targets {target!r}, "
                    "which has no hue"
                )

    return {
        "path": path,
        "dataset": dataset,
        "aliases": aliases,
        "class_id_to_name": class_id_to_name,
        "class_name_to_id": class_name_to_id,
        "hues": hues,
        "transformations": normalized_transformations,
    }


def canonical_class_name(class_name, aliases):
    """Resolve a dataset or annotation alias to its canonical semantic name."""
    return aliases.get(class_name, class_name)
