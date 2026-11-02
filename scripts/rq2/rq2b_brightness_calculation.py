#!/usr/bin/env python3
"""
RQ2.b: Brightness Scoring Method Calculation

RQ2.b: Among different methods of brightness scoring, which most reliably
predicts the degradation threshold identified in RQ2, and does the method
remain consistent across all models?

This script implements step 1 of RQ2.b: for the operational brightness
thresholds identified in RQ2.a (by default using the brute-force search
results; pass --search-method golden_section to use the golden-section
search results instead), compute a battery of brightness scores -- a mix of classic
methods (linear luminance, HSV/HSI, CIELAB L*, RMS contrast) and a novel
mean-shift clustering-based separation-to-spread ratio -- on the transformed
images at both the "left" (brighter) and "right" (darker) crossing gammas.

The core hypothesis being tested: if a brightness method reliably predicts
the point at which a model's accuracy degrades, then that method's score
should be roughly *the same* value at the crossing point for every image in
the dataset (i.e. low variance across the dataset). The method with the
lowest variance (or coefficient of variation, for a scale-independent
comparison) is therefore the most reliable "operational brightness"
predictor. Doing this separately per model checks whether the same method
remains the most reliable predictor across models.

Usage:
    # Auto-discover analysis/results/rq2/rq2a/brute_force/<model> results
    python scripts/rq2/rq2b_brightness_calculation.py

    # Use the golden-section search results instead of brute-force
    python scripts/rq2/rq2b_brightness_calculation.py --search-method golden_section

    # Restrict to specific models
    python scripts/rq2/rq2b_brightness_calculation.py --model yolov8 yolov5

    # Quick test on a handful of images
    python scripts/rq2/rq2b_brightness_calculation.py --num-images 10

    # Parallelize brightness scoring (mean-shift clustering is the bottleneck)
    python scripts/rq2/rq2b_brightness_calculation.py --workers 8
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.stats import spearmanr

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.model_registry import get_model_names
from utils.imgDarkBright_tuning.brightness_methods import (
    BRIGHTNESS_METHODS,
    calculate_all_brightness_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "analysis" / "results" / "rq2" / "rq2a"
ALL_SEARCH_METHODS = ["brute_force", "golden_section"]

DEFAULT_SEARCH_METHOD = "brute_force"
THRESHOLD_SIDES = ["left", "right"]  # left = brighter crossing, right = darker crossing

GAMMA_DIR_PATTERN = re.compile(r"^gamma_([0-9]*\.?[0-9]+)(?:_.*)?$")


def discover_results_files(models: List[str], search_methods: List[str]) -> List[Dict]:
    """
    Find all available RQ2.a results.json files for the given models.

    Args:
        models: Models to search for.
        search_methods: Which RQ2.a search method(s) results to look for
            (e.g. ["brute_force"] or ["golden_section"]).

    Returns:
        List of dicts: {"path": Path, "search_method": str, "model": str}
    """
    discovered = []
    for model in models:
        for search_method in search_methods:
            candidate = RESULTS_ROOT / search_method / model / "results.json"
            if candidate.is_file():
                discovered.append({
                    "path": candidate,
                    "search_method": search_method,
                    "model": model,
                })
    return discovered


def find_gamma_image(
    image_result_dir: Path,
    image_stem: str,
    target_gamma: float,
    tolerance: float = 0.05,
) -> Tuple[Optional[Path], Optional[float]]:
    """
    Locate the transformed image whose gamma value is (numerically) closest
    to ``target_gamma`` within an RQ2.a per-image result directory.

    RQ2.a saves one subdirectory per gamma value evaluated, named either
    ``gamma_<value>`` (grid points) or ``gamma_<value>_<label>`` (search
    refinement steps). We match on the leading numeric value rather than
    relying on a specific label, since the exact directory name used for the
    converged left/right crossing gamma can differ between the brute-force
    and golden-section scripts.

    Args:
        image_result_dir: Directory containing this image's gamma_* subdirs.
        image_stem: Image filename stem (without extension).
        target_gamma: The gamma_left or gamma_right value from results.json.
        tolerance: Max allowed |found_gamma - target_gamma| before warning.

    Returns:
        (path_to_transformed_image, gamma_diff) or (None, None) if not found.
    """
    if not image_result_dir.is_dir():
        return None, None

    best_path = None
    best_diff = float("inf")

    for entry in image_result_dir.iterdir():
        if not entry.is_dir():
            continue
        match = GAMMA_DIR_PATTERN.match(entry.name)
        if not match:
            continue

        gamma_value = float(match.group(1))
        diff = abs(gamma_value - target_gamma)
        if diff >= best_diff:
            continue

        matches = list(entry.glob(f"{image_stem}_imgDarkBright.*"))
        if not matches:
            continue

        best_diff = diff
        best_path = matches[0]

    return best_path, (best_diff if best_path is not None else None)


def _score_worker(args: Tuple[str, str, str, str, float, float]) -> Dict:
    """
    Worker function for parallel brightness scoring of one image record.

    Args (packed into a tuple for multiprocessing):
        search_method, model, image_name, image_result_dir, gamma_left, gamma_right
    """
    search_method, model, image_name, image_result_dir_str, gamma_left, gamma_right, ap_left, ap_right = args
    image_result_dir = Path(image_result_dir_str)
    image_stem = Path(image_name).stem

    record = {
        "search_method": search_method,
        "model": model,
        "image_name": image_name,
        "gamma_left": gamma_left,
        "gamma_right": gamma_right,
        "ap_left": ap_left,
        "ap_right": ap_right,
        "scores_left": None,
        "scores_right": None,
        "left_gamma_match_diff": None,
        "right_gamma_match_diff": None,
        "errors": [],
    }

    for side, target_gamma in (("left", gamma_left), ("right", gamma_right)):
        if target_gamma is None:
            continue

        image_path, diff = find_gamma_image(image_result_dir, image_stem, target_gamma)
        if image_path is None:
            record["errors"].append(f"No transformed image found for {side} gamma={target_gamma}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            record["errors"].append(f"Could not read image: {image_path}")
            continue

        record[f"scores_{side}"] = calculate_all_brightness_scores(image)
        record[f"{side}_gamma_match_diff"] = diff

    return record


def compute_score_records(
    discovered_files: List[Dict],
    num_images: Optional[int],
    workers: int,
) -> List[Dict]:
    """Load RQ2.a results and compute brightness scores at each threshold."""
    tasks = []

    for entry in discovered_files:
        path = entry["path"]
        search_method = entry["search_method"]
        model = entry["model"]

        with open(path) as f:
            data = json.load(f)

        results = data.get("results", [])
        if num_images:
            results = results[:num_images]

        print(f"Loaded {len(results)} image results from {path}")

        for image_result in results:
            gamma_left = image_result.get("gamma_left")
            gamma_right = image_result.get("gamma_right")
            if gamma_left is None and gamma_right is None:
                continue

            image_name = image_result["image_name"]
            image_result_dir = path.parent / Path(image_name).stem

            tasks.append((
                search_method,
                model,
                image_name,
                str(image_result_dir),
                gamma_left,
                gamma_right,
                image_result.get("ap_left"),
                image_result.get("ap_right"),
            ))

    print(f"\nScoring {len(tasks)} images across {len(discovered_files)} result file(s)...")

    records = []
    if workers > 1:
        with Pool(workers) as pool:
            for i, record in enumerate(pool.imap_unordered(_score_worker, tasks), start=1):
                records.append(record)
                if i % 10 == 0 or i == len(tasks):
                    print(f"  Scored {i}/{len(tasks)} images")
    else:
        for i, task in enumerate(tasks, start=1):
            records.append(_score_worker(task))
            if i % 10 == 0 or i == len(tasks):
                print(f"  Scored {i}/{len(tasks)} images")

    for record in records:
        for error in record["errors"]:
            print(f"  Warning [{record['search_method']}/{record['model']}/{record['image_name']}]: {error}")

    return records


def _collect_values(
    records: List[Dict],
    method_name: str,
    side: str,
    search_method: Optional[str] = None,
    model: Optional[str] = None,
) -> List[float]:
    """Gather all scores for one brightness method under the given filters."""
    values = []
    sides = ["left", "right"] if side == "combined" else [side]
    for record in records:
        if search_method is not None and record["search_method"] != search_method:
            continue
        if model is not None and record["model"] != model:
            continue
        for s in sides:
            scores = record.get(f"scores_{s}")
            if scores is not None and method_name in scores:
                values.append(scores[method_name])
    return values


def _stats_for_values(values: List[float]) -> Optional[Dict]:
    if not values:
        return None
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    variance = float(np.var(arr))
    cv = float(std / mean) if abs(mean) > 1e-12 else None
    return {
        "n": len(arr),
        "mean": mean,
        "std": std,
        "variance": variance,
        "coefficient_of_variation": cv,
    }


def build_method_statistics(records: List[Dict], models: List[str], search_methods: List[str]) -> Dict:
    """
    Compute per-method variance/CV statistics at multiple levels of
    aggregation: overall, per-model, and per-(search_method, model).
    """
    method_names = list(BRIGHTNESS_METHODS.keys())
    sides = THRESHOLD_SIDES + ["combined"]

    def ranked_stats(search_method, model):
        side_stats = {}
        for side in sides:
            method_stats = {}
            for method_name in method_names:
                values = _collect_values(records, method_name, side, search_method, model)
                stats = _stats_for_values(values)
                if stats is not None:
                    method_stats[method_name] = stats
            ranking_by_cv = sorted(
                (m for m, s in method_stats.items() if s["coefficient_of_variation"] is not None),
                key=lambda m: method_stats[m]["coefficient_of_variation"],
            )
            ranking_by_variance = sorted(
                method_stats.keys(),
                key=lambda m: method_stats[m]["variance"],
            )
            side_stats[side] = {
                "methods": method_stats,
                "ranking_by_coefficient_of_variation": ranking_by_cv,
                "ranking_by_raw_variance": ranking_by_variance,
            }
        return side_stats

    statistics = {
        "overall": ranked_stats(None, None),
        "by_model": {model: ranked_stats(None, model) for model in models},
        "by_search_method": {sm: ranked_stats(sm, None) for sm in search_methods},
        "by_search_method_and_model": {
            sm: {model: ranked_stats(sm, model) for model in models}
            for sm in search_methods
        },
    }
    return statistics


def compute_cross_model_consistency(statistics: Dict, models: List[str]) -> Dict:
    """
    Check whether the best (lowest-CV) brightness method is consistent
    across models, using Spearman rank correlation between each pair of
    models' method rankings, plus simple top-1 agreement.
    """
    method_names = list(BRIGHTNESS_METHODS.keys())
    consistency = {}

    for side in THRESHOLD_SIDES + ["combined"]:
        per_model_cv_rank = {}
        for model in models:
            method_stats = statistics["by_model"][model][side]["methods"]
            ranking = statistics["by_model"][model][side]["ranking_by_coefficient_of_variation"]
            rank_of = {m: r for r, m in enumerate(ranking)}
            # Methods missing CV get the worst possible rank
            per_model_cv_rank[model] = [
                rank_of.get(m, len(method_names)) for m in method_names
            ]

        pairwise_correlation = {}
        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                if not per_model_cv_rank[model_a] or not per_model_cv_rank[model_b]:
                    continue
                corr, pvalue = spearmanr(per_model_cv_rank[model_a], per_model_cv_rank[model_b])
                pairwise_correlation[f"{model_a}_vs_{model_b}"] = {
                    "spearman_r": float(corr) if corr is not None and not np.isnan(corr) else None,
                    "p_value": float(pvalue) if pvalue is not None and not np.isnan(pvalue) else None,
                }

        top_method_per_model = {
            model: (statistics["by_model"][model][side]["ranking_by_coefficient_of_variation"][0]
                    if statistics["by_model"][model][side]["ranking_by_coefficient_of_variation"] else None)
            for model in models
        }
        distinct_top_methods = set(m for m in top_method_per_model.values() if m is not None)

        consistency[side] = {
            "top_method_per_model": top_method_per_model,
            "consistent_top_method": len(distinct_top_methods) <= 1,
            "pairwise_spearman_correlation": pairwise_correlation,
            "overall_top_method": (
                statistics["overall"][side]["ranking_by_coefficient_of_variation"][0]
                if statistics["overall"][side]["ranking_by_coefficient_of_variation"] else None
            ),
        }

    return consistency


def print_summary(statistics: Dict, consistency: Dict, models: List[str]):
    print(f"\n{'='*80}")
    print("RQ2.b: Brightness Scoring Method Comparison")
    print(f"{'='*80}")

    for side in THRESHOLD_SIDES + ["combined"]:
        label = {"left": "LEFT crossing (brighter/over-exposed threshold)",
                  "right": "RIGHT crossing (darker/under-exposed threshold)",
                  "combined": "COMBINED (both crossings pooled)"}[side]
        print(f"\n{'-'*80}\n{label}\n{'-'*80}")

        overall = statistics["overall"][side]
        print(f"{'Method':<38}{'N':>6}{'Mean':>12}{'StdDev':>12}{'Variance':>14}{'CoefVar':>10}")
        for method_name in overall["ranking_by_coefficient_of_variation"]:
            s = overall["methods"][method_name]
            cv_str = f"{s['coefficient_of_variation']:.4f}" if s["coefficient_of_variation"] is not None else "N/A"
            print(f"{method_name:<38}{s['n']:>6}{s['mean']:>12.4f}{s['std']:>12.4f}{s['variance']:>14.6f}{cv_str:>10}")

        if overall["ranking_by_coefficient_of_variation"]:
            best = overall["ranking_by_coefficient_of_variation"][0]
            print(f"\n  Most reliable overall (lowest coefficient of variation): {best}")

        print(f"\n  Per-model top method (by coefficient of variation):")
        for model in models:
            top = consistency[side]["top_method_per_model"].get(model)
            print(f"    {model:<15}: {top if top else 'N/A'}")

        print(f"\n  Consistent across all models: {consistency[side]['consistent_top_method']}")
        if consistency[side]["pairwise_spearman_correlation"]:
            print("  Pairwise ranking correlation (Spearman r on CV-based method ranks):")
            for pair, corr in consistency[side]["pairwise_spearman_correlation"].items():
                r_str = f"{corr['spearman_r']:.3f}" if corr["spearman_r"] is not None else "N/A"
                print(f"    {pair:<30}: r={r_str}")

    print(f"\n{'='*80}\n")


def plot_method_comparison(statistics: Dict, output_dir: Path):
    """Bar chart of coefficient of variation per method, for the combined side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    side = "combined"
    overall = statistics["overall"][side]
    methods = [m for m in overall["ranking_by_coefficient_of_variation"]
               if m != "mean_shift_separation_spread_ratio"]
    if not methods:
        return

    cvs = [overall["methods"][m]["coefficient_of_variation"] for m in methods]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(methods)))
    ax.bar(range(len(methods)), cvs, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Brightness method consistency at Degradation Thresholds")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    output_path = output_dir / "method_comparison.pdf"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RQ2.b: Brightness Scoring Method Calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover available RQ2.a brute-force results and score every method
  python scripts/rq2/rq2b_brightness_calculation.py

  # Use golden-section search results instead
  python scripts/rq2/rq2b_brightness_calculation.py --search-method golden_section

  # Restrict to specific model
  python scripts/rq2/rq2b_brightness_calculation.py --model yolov8 yolov5

  # Quick smoke test
  python scripts/rq2/rq2b_brightness_calculation.py --num-images 5 --workers 4
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["yolov8"],
        help="Model to include (default: yolov8)",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        choices=ALL_SEARCH_METHODS,
        default=DEFAULT_SEARCH_METHOD,
        help="Which RQ2.a search method's results to use as the source of "
             f"operational brightness thresholds (default: {DEFAULT_SEARCH_METHOD}). "
             "Ignored if --results-json is passed explicitly.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of RQ2.a results.json paths to use instead of auto-discovery. "
             "Each file's 'method' and 'model' fields are used for grouping.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Limit number of images processed per results.json (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for brightness scoring (default: 1). "
             "Mean-shift clustering is the main bottleneck, so parallelizing helps a lot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: analysis/results/rq2/rq2b/brightness_scoring/<search_method>/<models>)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the comparison plot",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        results = Path("analysis/results/rq2")
        model_label = "_".join(args.model)
        output_dir = results / "rq2b" / "brightness_scoring" / args.search_method / model_label
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.results_json:
        discovered_files = []
        for path_str in args.results_json:
            path = Path(path_str)
            with open(path) as f:
                data = json.load(f)
            discovered_files.append({
                "path": path,
                "search_method": data.get("method", "unknown"),
                "model": data.get("model", "unknown"),
            })
        models = sorted(set(entry["model"] for entry in discovered_files))
        search_methods = sorted(set(entry["search_method"] for entry in discovered_files))
    else:
        models = args.model if args.model else get_model_names()
        search_methods = [args.search_method]
        discovered_files = discover_results_files(models, search_methods)
        # Restrict `models` to those that actually have discovered results
        models = sorted(set(entry["model"] for entry in discovered_files))

    if not discovered_files:
        print(f"Error: No RQ2.a results.json files found for search method(s) {search_methods}. "
              "Run rq2a_brute_force_search.py and/or rq2a_golden_section_search.py first, "
              "or pass --results-json explicitly.")
        sys.exit(1)

    print("="*80)
    print("RQ2.b: Brightness Scoring Method Calculation Configuration")
    print("="*80)
    print(f"Brightness methods ({len(BRIGHTNESS_METHODS)}): {', '.join(BRIGHTNESS_METHODS.keys())}")
    print(f"Models: {models}")
    print("Input files:")
    for entry in discovered_files:
        print(f"  [{entry['search_method']}] {entry['model']}: {entry['path']}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print("="*80)

    records = compute_score_records(discovered_files, args.num_images, args.workers)

    statistics = build_method_statistics(records, models, search_methods)
    consistency = compute_cross_model_consistency(statistics, models)

    print_summary(statistics, consistency, models)

    output_json = output_dir / "results.json"
    final_results = {
        "brightness_methods": list(BRIGHTNESS_METHODS.keys()),
        "models": models,
        "input_files": [
            {"path": str(entry["path"]), "search_method": entry["search_method"], "model": entry["model"]}
            for entry in discovered_files
        ],
        "num_images_scored": len(records),
        "per_image_results": records,
        "method_statistics": statistics,
        "cross_model_consistency": consistency,
    }

    with open(output_json, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to: {output_json}")

    if not args.no_plot:
        plot_method_comparison(statistics, output_dir)


if __name__ == "__main__":
    main()
