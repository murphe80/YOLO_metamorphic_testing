"""Source-specific hue plots shared by AP/AR reporting."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.model_registry import get_model_by_name


def plot_metric(settings, models, metric, comparison, output_path):
    # Match the confidence plot: one categorical position per included mapping,
    # ordered by hue setting, then by the reviewed source-class mapping order.
    cohorts = [
        (setting, mapping)
        for setting in settings
        for mapping in setting["included_mappings"]
    ]
    labels = [f"{setting['hue_rotation']}\n{mapping['mapping']}" for setting, mapping in cohorts]
    fig, axis = plt.subplots(figsize=(14, 7))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for index, model in enumerate(sorted(models)):
        values = [
            setting.get("class_shifts", {}).get(mapping["key"], {}).get("models", {}).get(model, {}).get(metric)
            for setting, mapping in cohorts
        ]
        config = get_model_by_name(model) or {}
        axis.plot(
            range(len(cohorts)),
            [float("nan") if value is None else value for value in values],
            marker=markers[index % len(markers)], linestyle="none", markersize=6,
            label=config.get("display_name", model),
            color=config.get("color", {}).get("hex"),
        )
    axis.set_xticks(range(len(cohorts)), labels, rotation=30, ha="right")
    axis.set_xlabel("Hue rotation / source → target (separate cohorts)", fontweight="bold")
    axis.set_ylabel(metric, fontweight="bold")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.3, linestyle=":", linewidth=0.8)
    axis.legend(loc="best", fontsize=9)
    excluded = [str(setting["hue_rotation"]) for setting in settings if not setting["included_mappings"]]
    title = f"Hue rotation — {metric} ({comparison.replace('_', ' ')})"
    if excluded:
        title += f"\nHue {', '.join(excluded)}: excluded (no included mappings)"
    axis.set_title(title, fontweight="bold", pad=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_metrics(variants, models, results_dir, comparison, metric):
    import json
    from analysis.utils.color_wheel import metadata
    settings = []
    for variant in variants:
        setting = metadata(variant["variant_id"])
        setting["setting"] = {"hue_rotation": setting["hue_rotation"]}
        setting["models"] = {}  # No pooled metric across source classes.
        setting["class_shifts"] = {
            m["key"]: {**m, "models": {}} for m in setting["included_mappings"]
        }
        for model in models:
            path = results_dir / model / variant["variant_id"] / comparison / "stats.json"
            stats = json.loads(path.read_text()) if path.exists() else {}
            if stats and "class_shifts" not in stats:
                raise ValueError(f"Regenerate color-wheel analysis before plotting: {path}")
            for key, shift in setting["class_shifts"].items():
                values = stats.get("class_shifts", {}).get(key, {})
                shift["models"][model] = {metric: values.get(metric), "AR": values.get("AR100")}
        settings.append(setting)
    return settings
