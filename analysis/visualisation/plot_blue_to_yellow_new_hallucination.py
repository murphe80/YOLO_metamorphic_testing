#!/usr/bin/env python3
"""Compare YOLO26 and YOLOv12 hallucinations for the blue-to-yellow cone rule.

Run from the repository root with:
    python3 analysis/visualisation/plot_blue_to_yellow_new_hallucination.py
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.visualisation.plot_detection_overlap_metrics import (
    METRICS,
    SIZE_BINS,
    Patch,
    get_model_display_name,
    load_json,
    metric_size_contributions,
    np,
    plt,
    size_color,
)


def main():
    results_root = PROJECT_ROOT / "analysis/results/rq1/test/detection_overlap_metrics"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics", type=Path,
        default=results_root / "detection_overlap_metrics.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=results_root / "plots/blue_to_yellow_new_hallucination_yolo26_yolov12.pdf",
    )
    args = parser.parse_args()
    statistics = load_json(args.statistics)
    models = ["yolov26_v2", "yolov12"]
    metric = "new_hallucination_rate"
    records = [statistics[f"{model}_blue_to_yellow"] for model in models]
    contributions = np.array([
        metric_size_contributions(record, metric) for record in records
    ])
    # Preserve the metric's original viridis colour and size-shading convention.
    color = plt.cm.viridis(np.linspace(0.2, 0.85, len(METRICS)))[METRICS.index(metric)]

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    for index, size in enumerate(SIZE_BINS):
        heights = contributions[:, index]
        ax.bar(
            x, heights, bottom=bottom, width=0.4,
            color=size_color(color, size), edgecolor="white", linewidth=0.4,
        )
        bottom += heights

    ax.set_xticks(x, [get_model_display_name(model) for model in models],
                  fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlim(-0.65, 1.65)
    ax.set_ylim(0.0, 0.4)
    ax.set_ylabel("Rate", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.suptitle("Blue to Yellow: New Hallucination Rate\nCone colour rule",
                 fontsize=15, fontweight="bold")
    fig.legend(
        [Patch(facecolor=size_color((0.5, 0.5, 0.5), size), edgecolor="white")
         for size in SIZE_BINS],
        ["Small (light)", "Medium", "Large (dark)"],
        loc="lower center", ncol=len(SIZE_BINS),
        title="Object size in transformed image", framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for model, record, total in zip(models, records, bottom):
        assert np.isclose(total, record[metric], rtol=1e-9, atol=1e-12)
        print(f"{get_model_display_name(model)}: {total:.8f} ({total:.2%})")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
