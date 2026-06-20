#!/usr/bin/env python3
"""
Metamorphic Configuration Tuning

Brute-force search for optimal metamorphic transformation configurations
that maximize model performance disparities between YOLOv5, v8, and v12.

Usage:
    # Default: All 6 transformations, 10 sandboxes, fsoco_sample dataset
    python scripts/tune_metamorphic_configs.py

    # Custom configuration
    python scripts/tune_metamorphic_configs.py \
        --num-sandboxes 10 \
        --dataset fsoco_sample \
        --transformations imgResolution imgColorWheel \
        --output-dir analysis/results/config_tuning

    # Production run
    python scripts/tune_metamorphic_configs.py \
        --dataset fsoco_test \
        --num-sandboxes 10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_generator import generate_configs, get_parameter_name
from utils.sandbox_manager import SandboxManager, ExperimentRunner
from utils.volume_helper import get_dataset_volume_id
from analysis.utils.config_analysis import (
    rank_configurations,
    export_optimal_configs,
    plot_disparity_curve,
    generate_summary_report,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConfigTuner:
    """Main orchestrator for configuration tuning experiments."""

    def __init__(
        self,
        api_key: str,
        num_sandboxes: int,
        dataset: str,
        transformations: List[str],
        output_dir: str,
        num_sample_images: int = 3,
    ):
        """
        Initialize Config Tuner.

        Args:
            api_key: Daytona API key
            num_sandboxes: Number of sandboxes to provision
            dataset: Dataset name (fsoco_sample or fsoco_test)
            transformations: List of transformation names to tune
            output_dir: Output directory for results
            num_sample_images: Number of sample images to save per config (default: 3)
        """
        self.api_key = api_key
        self.num_sandboxes = num_sandboxes
        self.dataset = dataset
        self.transformations = transformations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_sample_images = num_sample_images

        # Create logs directory for sample images
        self.samples_dir = Path("logs/config_samples")
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.sandbox_manager = None
        self.all_results = {}

        logger.info("=" * 80)
        logger.info("Metamorphic Configuration Tuning - Initialized")
        logger.info("=" * 80)
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Transformations: {', '.join(transformations)}")
        logger.info(f"Sandboxes: {num_sandboxes}")
        logger.info(f"Sample images per config: {num_sample_images}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Sample images: {self.samples_dir}")
        logger.info("=" * 80)

    def run(self):
        """Execute the full tuning workflow."""
        try:
            # Step 1: Provision sandboxes
            logger.info("\n[STEP 1/5] Provisioning Sandboxes")
            logger.info("=" * 80)
            # Daytona account limits: 10 CPU, 10GB RAM, 30GB disk
            # PyTorch + YOLO dependencies require ~8-10GB disk space
            # This limits us to 3 concurrent sandboxes (30GB / 10GB)
            actual_sandboxes = min(self.num_sandboxes, 3)
            if actual_sandboxes < self.num_sandboxes:
                logger.warning(
                    f"Requested {self.num_sandboxes} sandboxes, but limited to {actual_sandboxes} "
                    f"due to disk space constraints (30GB total, ~10GB needed per sandbox for ML dependencies)"
                )

            # Load dataset volume ID
            try:
                dataset_volume_id = get_dataset_volume_id()
                logger.info(f"Using dataset volume: {dataset_volume_id[:12]}...")
            except FileNotFoundError:
                logger.error("Dataset volume not found!")
                logger.error("Please run: python scripts/upload_dataset_to_volume.py")
                raise

            self.sandbox_manager = SandboxManager(
                api_key=self.api_key,
                num_sandboxes=actual_sandboxes,
                cpu=3,        # Total account limit is 10 CPU cores, 3 per sandbox for 3 sandboxes
                memory=3,     # Total account limit is 10GB RAM, 3GB per sandbox for 3 sandboxes
                disk=10,      # Need 10GB for PyTorch + YOLO dependencies
                gpu_count=0,  # No GPU access in current Daytona tier
                dataset_volume_id=dataset_volume_id,  # Mount dataset volume
            )
            self.sandbox_manager.provision_sandboxes()

            # Step 2: Setup sandboxes
            logger.info("\n[STEP 2/5] Setting Up Sandboxes")
            logger.info("=" * 80)
            self.sandbox_manager.setup_all_sandboxes(repo_path=".")

            # Step 3: Run experiments
            logger.info("\n[STEP 3/5] Running Experiments")
            logger.info("=" * 80)
            self._run_all_experiments()

            # Step 4: Analyze results
            logger.info("\n[STEP 4/5] Analyzing Results")
            logger.info("=" * 80)
            self._analyze_results()

            # Step 5: Generate reports
            logger.info("\n[STEP 5/5] Generating Reports")
            logger.info("=" * 80)
            self._generate_reports()

            logger.info("\n" + "=" * 80)
            logger.info("✓ Configuration Tuning Complete!")
            logger.info("=" * 80)
            logger.info(f"Results saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"✗ Tuning failed: {e}")
            raise

        finally:
            # Cleanup
            if self.sandbox_manager:
                logger.info("\n[CLEANUP] Deleting Sandboxes")
                logger.info("=" * 80)
                self.sandbox_manager.cleanup_sandboxes()

    def _run_single_experiment(self, sandbox, transformation, config, experiment_num, total_experiments):
        """Run a single experiment and return results."""
        variant_id = config["variant_id"]

        experiment_config = {
            "transformation": transformation,
            "variant_id": variant_id,
            "config": config,
        }

        logger.info(f"  [{experiment_num}/{total_experiments}] Running {variant_id}...")

        runner = ExperimentRunner(
            sandbox=sandbox,
            experiment_config=experiment_config,
            dataset=self.dataset,
            num_sample_images=self.num_sample_images,
        )

        scores = runner.run_experiment()

        # Download sample images for transparency logging
        try:
            local_paths = runner.download_sample_images(str(self.samples_dir))
            logger.info(f"  ✓ Sample images saved: {len(local_paths)} files")
        except Exception as e:
            logger.warning(f"  Failed to download samples: {e}")
            local_paths = []

        return {
            "variant_id": variant_id,
            "transformation": transformation,
            "config": config,
            "scores": scores,
            "sample_images": local_paths,
        }

    def _run_all_experiments(self):
        """
        Run experiments for all transformations.
        Process variants in parallel batches using available sandboxes.
        """
        total_experiments = len(self.transformations) * 10
        completed = 0
        num_sandboxes = len(self.sandbox_manager.sandboxes)

        for transformation in self.transformations:
            logger.info(f"\n{'─' * 80}")
            logger.info(f"Transformation: {transformation}")
            logger.info(f"{'─' * 80}")

            # Generate 10 config variants
            configs = generate_configs(transformation)
            logger.info(f"Generated {len(configs)} configuration variants")
            logger.info(f"Running in parallel batches of {num_sandboxes} sandboxes")

            start_time = time.time()

            # Process configs in batches equal to number of sandboxes
            for batch_start in range(0, len(configs), num_sandboxes):
                batch_configs = configs[batch_start:batch_start + num_sandboxes]

                logger.info(f"\n  Starting batch {batch_start//num_sandboxes + 1} "
                           f"({len(batch_configs)} experiments in parallel)...")

                # Run batch in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=num_sandboxes) as executor:
                    # Submit all experiments in this batch
                    futures = {}
                    for i, config in enumerate(batch_configs):
                        sandbox = self.sandbox_manager.sandboxes[i]
                        future = executor.submit(
                            self._run_single_experiment,
                            sandbox,
                            transformation,
                            config,
                            batch_start + i + 1,
                            len(configs)
                        )
                        futures[future] = config["variant_id"]

                    # Collect results as they complete
                    for future in as_completed(futures):
                        variant_id = futures[future]
                        try:
                            result = future.result()
                            self.all_results[result["variant_id"]] = result
                            completed += 1

                            elapsed = time.time() - start_time
                            logger.info(
                                f"  ✓ {result['variant_id']} complete | "
                                f"Progress: {completed}/{total_experiments} "
                                f"({completed/total_experiments*100:.1f}%) - "
                                f"Elapsed: {elapsed/60:.1f} min"
                            )
                        except Exception as e:
                            logger.error(f"  ✗ {variant_id} failed: {e}")
                            completed += 1

            logger.info(f"✓ {transformation} complete ({len(configs)} variants)")

    def _analyze_results(self):
        """Analyze and rank all results."""
        logger.info("Ranking configurations by disparity metric...")

        # Rank all configurations
        ranked_df = rank_configurations(self.all_results)

        # Save ranked table
        ranked_csv = self.output_dir / "ranked_configs.csv"
        ranked_df.to_csv(ranked_csv, index=False)
        logger.info(f"✓ Ranked configurations saved to: {ranked_csv}")

        # Export optimal configs
        optimal_json = self.output_dir.parent.parent / "config" / "optimal_transformations.json"
        export_optimal_configs(ranked_df, str(optimal_json), top_k=3)
        logger.info(f"✓ Optimal configurations saved to: {optimal_json}")

        # Store for reporting
        self.ranked_df = ranked_df

    def _generate_reports(self):
        """Generate visualizations and summary report."""
        # Create plots directory
        plots_dir = self.output_dir / "disparity_curves"
        plots_dir.mkdir(exist_ok=True)

        # Generate disparity curves for each transformation
        for transformation in self.transformations:
            logger.info(f"  Plotting disparity curve: {transformation}")

            # Filter results for this transformation
            transform_results = [
                {
                    "config": data["config"],
                    "scores": data["scores"],
                }
                for variant_id, data in self.all_results.items()
                if data["transformation"] == transformation
            ]

            param_name = get_parameter_name(transformation)
            plot_path = plots_dir / f"{transformation}.png"

            plot_disparity_curve(
                transformation=transformation,
                results=transform_results,
                parameter_name=param_name,
                output_path=str(plot_path),
            )

        logger.info(f"✓ Disparity curves saved to: {plots_dir}")

        # Generate summary report
        report_path = self.output_dir / "summary_report.md"
        generate_summary_report(self.ranked_df, str(report_path))
        logger.info(f"✓ Summary report saved to: {report_path}")

        # Save raw results as JSON
        results_json = self.output_dir / "experiment_results.json"
        with open(results_json, "w") as f:
            json.dump(self.all_results, f, indent=2)
        logger.info(f"✓ Raw results saved to: {results_json}")

        # Generate sample images index
        self._generate_samples_index()
        logger.info(f"✓ Sample images index saved to: {self.samples_dir / 'INDEX.md'}")

    def _generate_samples_index(self):
        """Generate an index file documenting all sample images."""
        index_lines = []
        index_lines.append("# Configuration Sample Images Index\n")
        index_lines.append("Sample transformed images for transparency and verification.\n")
        index_lines.append(f"**Dataset:** {self.dataset}\n")
        index_lines.append(f"**Samples per configuration:** {self.num_sample_images}\n\n")

        # Group by transformation
        for transformation in self.transformations:
            index_lines.append(f"## {transformation}\n")

            # Get all variants for this transformation
            variants = [
                (variant_id, data)
                for variant_id, data in self.all_results.items()
                if data["transformation"] == transformation
            ]

            # Sort by variant_id
            variants.sort(key=lambda x: x[0])

            for variant_id, data in variants:
                config = data["config"]
                sample_images = data.get("sample_images", [])

                index_lines.append(f"\n### {variant_id}\n")
                index_lines.append(f"**Configuration:** `{json.dumps(config)}`\n")
                index_lines.append(f"**Sample images:** {len(sample_images)}\n\n")

                if sample_images:
                    for img_path in sample_images:
                        rel_path = Path(img_path).relative_to(self.samples_dir.parent)
                        index_lines.append(f"- `{rel_path}`\n")
                else:
                    index_lines.append("- *(No samples available)*\n")

            index_lines.append("\n")

        # Write index
        index_file = self.samples_dir / "INDEX.md"
        with open(index_file, "w") as f:
            f.write("\n".join(index_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Metamorphic Configuration Tuning with Daytona Sandboxes"
    )
    parser.add_argument(
        "--num-sandboxes",
        type=int,
        default=10,
        help="Number of Daytona sandboxes to provision (default: 10)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fsoco_test",
        choices=["fsoco_sample", "fsoco_test"],
        help="Dataset to use for experiments (default: fsoco_test)",
    )
    parser.add_argument(
        "--transformations",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "imgColorWheel",
            "imgResolution",
            "imgDarkBright",
            "imgMotionBlur",
            "imgStretch",
        ],
        help="Transformations to tune (default: all 6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/results/config_tuning",
        help="Output directory for results (default: analysis/results/config_tuning)",
    )
    parser.add_argument(
        "--num-sample-images",
        type=int,
        default=3,
        help="Number of sample images to save per config for logging (default: 3)",
    )

    args = parser.parse_args()

    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("DAYTONA_API_KEY")

    if not api_key:
        logger.error("Error: DAYTONA_API_KEY not found in environment")
        logger.error("Please set it in .env file or environment variables")
        sys.exit(1)

    # Default to all transformations if not specified
    if args.transformations is None:
        args.transformations = [
            "imgColorWheel",
            "imgResolution",
            "imgDarkBright",
            "imgMotionBlur",
            "imgStretch",
        ]

    # Initialize and run tuner
    tuner = ConfigTuner(
        api_key=api_key,
        num_sandboxes=args.num_sandboxes,
        dataset=args.dataset,
        transformations=args.transformations,
        output_dir=args.output_dir,
        num_sample_images=args.num_sample_images,
    )

    tuner.run()


if __name__ == "__main__":
    main()
