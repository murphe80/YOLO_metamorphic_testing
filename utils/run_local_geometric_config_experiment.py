#!/usr/bin/env python3
"""
Run local sequential configuration tuning for imgResolution and imgStretch.

This mirrors the Daytona ExperimentRunner workflow, but executes each
configuration locally:
  1. Generate transformed images.
  2. Run inference for each selected model.
  3. Run original-vs-transformed and transformed-vs-GT analysis.
  4. Rank configurations by model AP50 disparity.

For geometric transformations, transformed detections are mapped back to the
original image coordinate system before GT matching.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.utils.format_detections import format_detections
from utils.config_generator import generate_configs, get_parameter_name
from utils.matias_metamorphic import gen_metamorphic
from utils.model_registry import get_model_by_name


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GEOMETRIC_TRANSFORMATIONS = ("imgResolution",)
COCO_STAT_KEYS = [
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
]


def image_files(image_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """Return image files in stable order."""
    files = sorted(
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    return files[:limit] if limit else files


def remove_variant_id(config: Dict) -> Dict:
    """Return a transformation config without tuning metadata."""
    cleaned = dict(config)
    cleaned.pop("variant_id", None)
    return cleaned


def transformed_name(image_path: Path, transformation: str) -> str:
    """Build the filename created by matias_metamorphic.py."""
    return f"{image_path.stem}_{transformation}{image_path.suffix}"


def run_command(cmd: List[str], cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess and stream output through the parent process."""
    logger.debug("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def write_stats(stats, stats_path: Path) -> Dict[str, float]:
    """Persist pycocotools stats with stable key names."""
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_dict = dict(zip(COCO_STAT_KEYS, [float(x) for x in stats]))
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    return stats_dict


def evaluate_and_save(gt_file: Path, dt_file: Path, stats_path: Path) -> Dict[str, float]:
    """Run COCO evaluation and save stats.json."""
    try:
        from analysis.coco_eval import evaluate_coco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "COCO evaluation requires pycocotools. Install project requirements "
            "before running analysis: python3 -m pip install -r requirements.txt"
        ) from exc

    stats = evaluate_coco(str(gt_file), str(dt_file))
    return write_stats(stats, stats_path)


def bbox_back_to_original(bbox: List[float], transformation: str, config: Dict) -> List[float]:
    """Map a transformed-image bbox back into the original image coordinate system."""
    x, y, w, h = bbox
    if transformation == "imgResolution":
        ratio = float(config["size_ratio"])
        scale_x = 1.0 / ratio
        scale_y = 1.0 / ratio
    elif transformation == "imgStretch":
        scale_x = 1.0 / float(config["width_ratio"])
        scale_y = 1.0 / float(config["height_ratio"])
    else:
        scale_x = scale_y = 1.0

    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]


def transform_detections_back_to_original(
    detection_file: Path,
    output_file: Path,
    transformation: str,
    config: Dict,
) -> Path:
    """
    Transform detections from imgResolution/imgStretch space to original space.

    This intentionally uses the exact tuning config rather than defaults or image
    dimension inference, so GT matching is correct for variant directories such
    as imgResolution_4 and imgStretch_7.
    """
    with open(detection_file) as f:
        detections = json.load(f)

    if isinstance(detections, dict):
        annotations = detections.get("annotations", [])
    else:
        annotations = detections

    transformed_annotations = []
    for ann in annotations:
        out = ann.copy()
        out["bbox"] = bbox_back_to_original(out["bbox"], transformation, config)
        out["area"] = out["bbox"][2] * out["bbox"][3]
        transformed_annotations.append(out)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(detections, dict):
        output_data = detections.copy()
        output_data["annotations"] = transformed_annotations
    else:
        output_data = transformed_annotations

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Transformed %d detections back to original coordinates: %s", len(transformed_annotations), output_file)
    return output_file


class LocalGeometricExperimentRunner:
    """Runs one local imgResolution/imgStretch tuning variant."""

    def __init__(
        self,
        transformation: str,
        config: Dict,
        dataset: str,
        models: Iterable[str],
        image_limit: Optional[int],
        num_sample_images: int,
        overwrite: bool,
        device: str,
        conf_thres: float,
        iou_thres: float,
    ):
        self.transformation = transformation
        self.config = remove_variant_id(config)
        self.variant_id = config["variant_id"]
        self.dataset = dataset
        self.models = list(models)
        self.image_limit = image_limit
        self.num_sample_images = num_sample_images
        self.overwrite = overwrite
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.images_dir = ROOT / "data" / "base_images" / dataset / "images"
        self.gt_coco = ROOT / "data" / "base_images" / dataset / "annotations_coco.json"
        self.output_dir = ROOT / "data" / "transformed_images" / dataset / self.variant_id
        self.samples_dir = ROOT / "logs" / "config_samples" / self.variant_id

    def run_experiment(self) -> Dict:
        """Run the full local experiment and return collected AP50 scores."""
        logger.info("Running experiment: %s", self.variant_id)
        # self._generate_transformed_images()
        # self._run_inference_all_models()
        scores = self._run_analysis()
        logger.info("Completed %s with AP50 scores: %s", self.variant_id, scores["original_vs_transformed"])
        return {
            "variant_id": self.variant_id,
            "transformation": self.transformation,
            "config": {**self.config, "variant_id": self.variant_id},
            "scores": scores["original_vs_transformed"],
            "gt_scores": scores["transformed_vs_gt"],
            "sample_images": [str(p) for p in sorted(self.samples_dir.glob("*"))],
        }

    def _generate_transformed_images(self):
        """Generate transformed images using matias_metamorphic.py logic."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")

        if self.overwrite and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        if self.overwrite and self.samples_dir.exists():
            shutil.rmtree(self.samples_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        images = image_files(self.images_dir, self.image_limit)
        if not images:
            raise RuntimeError(f"No input images found in {self.images_dir}")

        for i, image_path in enumerate(images):
            output_path = self.output_dir / transformed_name(image_path, self.transformation)
            if output_path.exists() and not self.overwrite:
                success = True
            else:
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.warning("Could not read image: %s", image_path)
                    continue
                success, _ = gen_metamorphic(
                    img,
                    self.transformation,
                    self.config,
                    (image_path.stem, image_path.suffix),
                    str(self.output_dir),
                )

            if success and i < self.num_sample_images:
                sample_path = self.samples_dir / f"sample_{i:03d}_{output_path.name}"
                if output_path.exists() and (self.overwrite or not sample_path.exists()):
                    shutil.copy2(output_path, sample_path)

    def _run_inference_all_models(self):
        """Run inference for selected models on original and transformed images."""
        original_images = image_files(self.images_dir, self.image_limit)
        transformed_images = image_files(self.output_dir, self.image_limit)
        if not transformed_images:
            raise RuntimeError(f"No transformed images found in {self.output_dir}")

        for model_name in self.models:
            model = get_model_by_name(model_name)
            if model is None:
                raise ValueError(f"Unknown model: {model_name}")

            script = ROOT / model["inference_script"]
            weights = ROOT / model["weights"]
            if not script.exists():
                raise FileNotFoundError(f"Inference script not found for {model_name}: {script}")
            if not weights.exists():
                raise FileNotFoundError(f"Weights not found for {model_name}: {weights}")

            logger.info("Running %s inference on original images", model_name)
            self._run_model_on_images(model_name, script, weights, original_images, "original")

            logger.info("Running %s inference on %s", model_name, self.variant_id)
            self._run_model_on_images(model_name, script, weights, transformed_images, self.variant_id)

    def _run_model_on_images(
        self,
        model_name: str,
        script: Path,
        weights: Path,
        images: List[Path],
        output_subdir: str,
    ):
        results_dir = ROOT / "results" / model_name / output_subdir
        results_dir.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            out_txt = results_dir / f"{image_path.stem}.txt"
            if out_txt.exists() and not self.overwrite:
                continue
            run_command(
                [
                    sys.executable,
                    str(script),
                    "--weights",
                    str(weights),
                    "--image",
                    str(image_path),
                    "--output-subdir",
                    output_subdir,
                    "--device",
                    self.device,
                    "--conf_thres",
                    str(self.conf_thres),
                    "--iou_thres",
                    str(self.iou_thres),
                ]
            )

    def _run_analysis(self) -> Dict[str, Dict[str, float]]:
        """Run original-vs-transformed and transformed-vs-GT analysis."""
        scores = {
            "original_vs_transformed": {},
            "transformed_vs_gt": {},
        }

        for model_name in self.models:
            model_results = ROOT / "results" / model_name
            original_dir = model_results / "original"
            variant_dir = model_results / self.variant_id
            stats_dir_base = ROOT / "analysis" / "results" / model_name / self.variant_id

            base_det = original_dir / "detections_original_coco.json"
            transformed_det = variant_dir / "detections_coco.json"
            transformed_back_det = variant_dir / "detections_coco_original_coords.json"

            format_detections(str(original_dir), str(base_det), as_ground_truth=True)
            format_detections(str(variant_dir), str(transformed_det))
            transform_detections_back_to_original(
                transformed_det,
                transformed_back_det,
                self.transformation,
                self.config,
            )

            logger.info("Comparing %s original vs %s", model_name, self.variant_id)
            metamorphic_stats = evaluate_and_save(
                base_det,
                transformed_back_det,
                stats_dir_base / "original_vs_transformed" / "stats.json",
            )
            scores["original_vs_transformed"][model_name] = metamorphic_stats.get("AP50", 0.0)

            if self.gt_coco.exists():
                gt_det = variant_dir / "detections_for_gt.json"
                gt_back_det = variant_dir / "detections_for_gt_original_coords.json"
                format_detections(str(variant_dir), str(gt_det), gt_coco_file=str(self.gt_coco))
                transform_detections_back_to_original(
                    gt_det,
                    gt_back_det,
                    self.transformation,
                    self.config,
                )

                logger.info("Comparing %s %s vs ground truth", model_name, self.variant_id)
                gt_stats = evaluate_and_save(
                    self.gt_coco,
                    gt_back_det,
                    stats_dir_base / "transformed_vs_gt" / "stats.json",
                )
                scores["transformed_vs_gt"][model_name] = gt_stats.get("AP50", 0.0)
            else:
                logger.warning("Ground truth not found, skipping transformed_vs_gt: %s", self.gt_coco)
                scores["transformed_vs_gt"][model_name] = 0.0

        return scores


def rank_results(all_results: Dict[str, Dict], score_source: str) -> List[Dict]:
    """Rank variants by AP50 disparity for the chosen score source."""
    ranked = []
    for variant_id, data in all_results.items():
        scores = data["gt_scores"] if score_source == "transformed_vs_gt" else data["scores"]
        ranked.append(
            {
                "variant_id": variant_id,
                "transformation": data["transformation"],
                "config": data["config"],
                "scores": scores,
                "original_vs_transformed_scores": data["scores"],
                "transformed_vs_gt_scores": data["gt_scores"],
                "disparity": compute_disparity_metric(scores),
            }
        )
    ranked.sort(key=lambda row: row["disparity"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def compute_disparity_metric(ap_scores: Dict[str, float]) -> float:
    """Compute max pairwise AP50 difference across models with available scores."""
    scores = [float(score) for score in ap_scores.values() if float(score) > 0.0]
    if len(scores) < 2:
        return 0.0
    return max(abs(left - right) for i, left in enumerate(scores) for right in scores[i + 1 :])


def write_ranked_csv(ranked: List[Dict], output_path: Path, models: List[str]):
    """Write a compact ranked CSV without requiring pandas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["rank", "variant_id", "transformation", "disparity", *[f"{m}_ap50" for m in models], "config"]
    rows = [",".join(header)]
    for row in ranked:
        values = [
            str(row["rank"]),
            row["variant_id"],
            row["transformation"],
            f"{row['disparity']:.6f}",
            *[f"{row['scores'].get(m, 0.0):.6f}" for m in models],
            json.dumps(row["config"]).replace('"', '""'),
        ]
        values[-1] = f'"{values[-1]}"'
        rows.append(",".join(values))
    output_path.write_text("\n".join(rows) + "\n")


def write_summary(ranked: List[Dict], output_path: Path, score_source: str):
    """Write a Markdown summary with the best config per transformation."""
    lines = [
        "# Local Geometric Configuration Tuning",
        "",
        f"Score source: `{score_source}`",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Variant | Transformation | Disparity | Scores | Config |",
        "|---:|---|---|---:|---|---|",
    ]

    for row in ranked:
        lines.append(
            "| {rank} | {variant_id} | {transformation} | {disparity:.4f} | `{scores}` | `{config}` |".format(
                rank=row["rank"],
                variant_id=row["variant_id"],
                transformation=row["transformation"],
                disparity=row["disparity"],
                scores=json.dumps(row["scores"]),
                config=json.dumps(row["config"]),
            )
        )

    lines.extend(["", "## Best Per Transformation", ""])
    seen = set()
    for row in ranked:
        if row["transformation"] in seen:
            continue
        seen.add(row["transformation"])
        lines.append(f"- `{row['transformation']}`: `{row['variant_id']}` with disparity `{row['disparity']:.4f}`")

    output_path.write_text("\n".join(lines) + "\n")


def save_results(
    all_results: Dict[str, Dict],
    output_dir: Path,
    models: List[str],
    score_source: str,
) -> List[Dict]:
    """Save raw and ranked outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "experiment_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)

    ranked = rank_results(all_results, score_source)
    with open(output_dir / "ranked_configs.json", "w") as f:
        json.dump(ranked, f, indent=2)

    write_ranked_csv(ranked, output_dir / "ranked_configs.csv", models)
    write_summary(ranked, output_dir / "summary_report.md", score_source)

    optimal = {}
    for row in ranked:
        optimal.setdefault(row["transformation"], [])
        if len(optimal[row["transformation"]]) < 3:
            optimal[row["transformation"]].append(row)
    with open(output_dir / "optimal_transformations.json", "w") as f:
        json.dump(optimal, f, indent=2)

    logger.info("Saved raw results to %s", raw_path)
    if ranked:
        logger.info(
            "Best config: %s disparity=%.4f scores=%s",
            ranked[0]["variant_id"],
            ranked[0]["disparity"],
            ranked[0]["scores"],
        )
    return ranked


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local sequential imgResolution/imgStretch configuration tuning"
    )
    parser.add_argument("--dataset", default="test", help="Dataset under data/base_images (default: fsoco_test)")
    parser.add_argument(
        "--transformations",
        nargs="+",
        default=list(GEOMETRIC_TRANSFORMATIONS),
        choices=list(GEOMETRIC_TRANSFORMATIONS),
        help="Geometric transformations to tune",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov5", "yolov8", "yolov12"],
        help="Model names from config/models.json (default: yolov5 yolov8 yolov12)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/results/local_geometric_config_tuning",
        help="Directory for tuning result summaries",
    )
    parser.add_argument("--image-limit", type=int, default=None, help="Optional limit for quick local runs")
    parser.add_argument("--num-sample-images", type=int, default=3, help="Sample transformed images to copy per config")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate images and rerun existing inference outputs")
    parser.add_argument("--device", default="cuda:0", help="Torch device passed to inference scripts")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Inference confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="Inference IoU threshold")
    parser.add_argument(
        "--score-source",
        choices=["original_vs_transformed", "transformed_vs_gt"],
        default="original_vs_transformed",
        help="Score set used for final disparity ranking (default matches sandbox manager)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = {}

    logger.info("Local geometric configuration tuning")
    logger.info("Dataset: %s", args.dataset)
    logger.info("Transformations: %s", ", ".join(args.transformations))
    logger.info("Models: %s", ", ".join(args.models))
    logger.info("Score source: %s", args.score_source)

    total = sum(len(generate_configs(t)) for t in args.transformations)
    completed = 0

    for transformation in args.transformations:
        configs = generate_configs(transformation)
        param = get_parameter_name(transformation)
        logger.info("Generated %d configs for %s (varying %s)", len(configs), transformation, param)

        for config in configs:
            completed += 1
            logger.info("[%d/%d] Starting %s", completed, total, config["variant_id"])
            runner = LocalGeometricExperimentRunner(
                transformation=transformation,
                config=config,
                dataset=args.dataset,
                models=args.models,
                image_limit=args.image_limit,
                num_sample_images=args.num_sample_images,
                overwrite=args.overwrite,
                device=args.device,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
            )
            result = runner.run_experiment()
            all_results[result["variant_id"]] = result
            save_results(all_results, ROOT / args.output_dir, args.models, args.score_source)

    save_results(all_results, ROOT / args.output_dir, args.models, args.score_source)


if __name__ == "__main__":
    main()
