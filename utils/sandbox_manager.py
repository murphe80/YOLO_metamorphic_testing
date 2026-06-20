#!/usr/bin/env python3
"""
Sandbox Manager for Daytona-based Experiment Execution

Provisions GPU-enabled Daytona sandboxes, sets up the experiment environment,
runs metamorphic testing experiments, and collects results.

Usage:
    from utils.sandbox_manager import SandboxManager, ExperimentRunner

    manager = SandboxManager(api_key=DAYTONA_API_KEY, num_sandboxes=10)
    manager.provision_sandboxes()
    manager.setup_all_sandboxes()

    runner = ExperimentRunner(sandbox, experiment_config)
    results = runner.run_experiment()
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import os

from daytona_sdk import (
    Daytona,
    Sandbox,
    CreateSandboxFromImageParams,
    Image,
    Resources,
    DaytonaConfig,
    VolumeMount,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages provisioning and lifecycle of Daytona sandboxes for experiments."""

    def __init__(
        self,
        api_key: str,
        num_sandboxes: int = 10,
        gpu_count: int = 0,
        cpu: int = 4,
        memory: int = 8,
        disk: int = 10,
        dataset_volume_id: str = None,
    ):
        """
        Initialize Sandbox Manager.

        Args:
            api_key: Daytona API key
            num_sandboxes: Number of sandboxes to provision (default: 10)
            gpu_count: GPUs per sandbox (default: 0, CPU-only)
            cpu: CPU cores per sandbox (default: 4)
            memory: Memory in GiB per sandbox (default: 8, max: 8)
            disk: Disk space in GiB per sandbox (default: 10, max: 10)
            dataset_volume_id: Optional Daytona volume ID containing dataset (default: None)
        """
        config = DaytonaConfig(api_key=api_key)
        self.daytona = Daytona(config=config)
        self.num_sandboxes = num_sandboxes
        self.gpu_count = gpu_count
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.dataset_volume_id = dataset_volume_id
        self.sandboxes: List[Sandbox] = []

        volume_status = f"Volume: {dataset_volume_id[:12]}..." if dataset_volume_id else "No volume"
        logger.info(
            f"Initialized SandboxManager: {num_sandboxes} sandboxes, "
            f"Resources: {cpu} CPU, {memory} GiB RAM, {disk} GiB disk, {gpu_count} GPU, "
            f"{volume_status}"
        )

    def provision_sandboxes(self, max_retries: int = 3) -> List[Sandbox]:
        """
        Provision all sandboxes with optional dataset volume.

        Args:
            max_retries: Max retry attempts for failed provisioning (default: 3)

        Returns:
            List of provisioned Sandbox objects

        Raises:
            RuntimeError: If sandbox provisioning fails after retries
        """
        logger.info(f"Provisioning {self.num_sandboxes} sandboxes...")

        resources = Resources(
            cpu=self.cpu,
            memory=self.memory,
            disk=self.disk,
            gpu=self.gpu_count,
        )

        # Configure volume mount if dataset volume is provided
        volumes = []
        if self.dataset_volume_id:
            volume_mount = VolumeMount(
                volume_id=self.dataset_volume_id,
                mount_path="/mnt/dataset"
            )
            volumes.append(volume_mount)
            logger.info(f"Dataset volume will be mounted at /mnt/dataset")

        for i in range(self.num_sandboxes):
            for attempt in range(max_retries):
                try:
                    logger.info(f"Creating sandbox {i+1}/{self.num_sandboxes}...")

                    # GPU sandboxes must be ephemeral (auto_delete_interval=0)
                    params = CreateSandboxFromImageParams(
                        image=Image.debian_slim("3.10"),
                        language="python",
                        resources=resources,
                        volumes=volumes if volumes else None,
                    )

                    if self.gpu_count > 0:
                        params.auto_delete_interval = 0

                    sandbox = self.daytona.create(params=params)

                    # Label for tracking
                    sandbox.set_labels(
                        {
                            "role": "config_tuner",
                            "index": str(i),
                            "framework": "yolo_metamorphic",
                        }
                    )

                    self.sandboxes.append(sandbox)
                    logger.info(f"✓ Sandbox {i+1} created: {sandbox.id}")
                    break

                except Exception as e:
                    logger.warning(
                        f"Sandbox {i+1} provisioning attempt {attempt+1} failed: {e}"
                    )
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Failed to provision sandbox {i+1} after {max_retries} attempts"
                        )
                    time.sleep(2 ** attempt)  # Exponential backoff

        logger.info(f"✓ All {len(self.sandboxes)} sandboxes provisioned successfully")
        return self.sandboxes

    def setup_sandbox(self, sandbox: Sandbox, repo_path: str = "."):
        """
        Set up a single sandbox with repo, dependencies, and model weights.

        Args:
            sandbox: Sandbox object to set up
            repo_path: Local path to repo root (default: current directory)

        Raises:
            RuntimeError: If setup fails
        """
        logger.info(f"Setting up sandbox {sandbox.id}...")

        try:
            # Get work directory
            work_dir = sandbox.get_work_dir()
            logger.info(f"  Work directory: {work_dir}")

            # Upload repository files (excluding large files and venv)
            logger.info("  Uploading repository files...")
            exclude_patterns = [
                "venv",
                ".git",
                "*.pyc",
                "__pycache__",
                "*.pt",  # Don't upload model weights initially
                "data/base_images",  # Will sync only needed dataset
                "results",
                "analysis/results",
                ".DS_Store",
            ]

            # Upload critical files
            critical_files = [
                "utils",
                "models",
                "analysis",
                "scripts",
                "config",
                "requirements.txt",
                ".env",
            ]

            for item in critical_files:
                src_path = Path(repo_path) / item
                if src_path.exists():
                    dest_path = f"{work_dir}/{item}"
                    logger.info(f"    Uploading {item}...")

                    if src_path.is_file():
                        # Upload file directly
                        sandbox.fs.upload_file(str(src_path), dest_path)
                    else:
                        # For directories, create tar and upload
                        import tarfile
                        import tempfile

                        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                            tar_path = tmp.name

                        try:
                            # Create tar
                            with tarfile.open(tar_path, 'w:gz') as tar:
                                tar.add(str(src_path), arcname=item)

                            # Upload tar
                            sandbox.fs.upload_file(tar_path, f"{work_dir}/{item}.tar.gz")

                            # Extract in sandbox
                            sandbox.process.exec(
                                f"cd {work_dir} && tar -xzf {item}.tar.gz && rm {item}.tar.gz",
                                timeout=60000
                            )
                        finally:
                            # Cleanup local tar
                            Path(tar_path).unlink(missing_ok=True)

            # Setup dataset symlink if volume is mounted
            if self.dataset_volume_id:
                logger.info("  Setting up dataset symlink from volume...")
                sandbox.process.exec(f"mkdir -p {work_dir}/data", timeout=5000)
                sandbox.process.exec(
                    f"ln -s /mnt/dataset {work_dir}/data/base_images",
                    timeout=5000
                )
                # Verify symlink
                result = sandbox.process.exec(f"ls -la {work_dir}/data/base_images", timeout=5000)
                if result.exit_code == 0:
                    logger.info("  ✓ Dataset symlink created from volume")
                else:
                    logger.warning(f"  ⚠ Symlink verification failed: {result.result}")

            # Install system dependencies for OpenCV
            logger.info("  Installing system dependencies for OpenCV...")
            result = sandbox.process.exec(
                "apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev",
                timeout=120000,  # 2 minutes
            )
            if result.exit_code == 0:
                logger.info("  ✓ System dependencies installed")
            else:
                logger.warning(f"  ⚠ System dependency installation had issues: {result.result[:200]}")

            # Install dependencies
            logger.info("  Installing Python dependencies...")
            result = sandbox.process.exec(
                f"cd {work_dir} && pip install --no-cache-dir -q -r requirements.txt",
                timeout=300000,  # 5 minutes
            )
            if result.exit_code != 0:
                logger.warning(f"  Dependency installation had issues: {result.result[:500]}")

            # Verify GPU availability
            logger.info("  Verifying GPU access...")
            gpu_check = sandbox.process.exec("nvidia-smi", timeout=10000)
            if gpu_check.exit_code == 0:
                logger.info("  ✓ GPU detected")
            else:
                logger.warning("  ⚠ GPU not detected - inference will be slow")

            logger.info(f"✓ Sandbox {sandbox.id} setup complete")

        except Exception as e:
            raise RuntimeError(f"Failed to setup sandbox {sandbox.id}: {e}")

    def setup_all_sandboxes(self, repo_path: str = "."):
        """
        Set up all provisioned sandboxes in parallel.

        Args:
            repo_path: Local path to repo root
        """
        logger.info(f"Setting up {len(self.sandboxes)} sandboxes...")

        # Sequential setup for simplicity (could be parallelized)
        for i, sandbox in enumerate(self.sandboxes):
            logger.info(f"Setting up sandbox {i+1}/{len(self.sandboxes)}...")
            self.setup_sandbox(sandbox, repo_path)

        logger.info("✓ All sandboxes setup complete")

    def cleanup_sandboxes(self):
        """Stop and delete all provisioned sandboxes."""
        logger.info(f"Cleaning up {len(self.sandboxes)} sandboxes...")

        for i, sandbox in enumerate(self.sandboxes):
            try:
                logger.info(f"Deleting sandbox {i+1}/{len(self.sandboxes)}: {sandbox.id}")
                sandbox.delete(timeout=60)
                logger.info(f"  ✓ Deleted")
            except Exception as e:
                logger.warning(f"  Failed to delete sandbox {sandbox.id}: {e}")

        self.sandboxes.clear()
        logger.info("✓ Cleanup complete")


class ExperimentRunner:
    """Runs a single metamorphic testing experiment in a Daytona sandbox."""

    def __init__(
        self,
        sandbox: Sandbox,
        experiment_config: Dict,
        dataset: str = "fsoco_sample",
        timeout: int = 1800000,  # 30 minutes in milliseconds
        num_sample_images: int = 3,
    ):
        """
        Initialize Experiment Runner.

        Args:
            sandbox: Daytona sandbox to run experiment in
            experiment_config: Dict with transformation, variant_id, config
            dataset: Dataset name (fsoco_sample or fsoco_test)
            timeout: Timeout per experiment in milliseconds (default: 30 min)
            num_sample_images: Number of sample images to save for logging (default: 3)
        """
        self.sandbox = sandbox
        self.config = experiment_config
        self.dataset = dataset
        self.timeout = timeout
        self.num_sample_images = num_sample_images
        self.work_dir = sandbox.get_work_dir()

        self.transformation = experiment_config["transformation"]
        self.variant_id = experiment_config["variant_id"]
        self.transform_config = experiment_config["config"]
        self.sample_images = []  # Track saved sample images

    def run_experiment(self) -> Dict[str, float]:
        """
        Run full metamorphic testing experiment.

        Workflow:
        1. Generate transformed images (with samples for logging)
        2. Run inference (yolov5, yolov8, yolov12)
        3. Run analysis
        4. Extract AP50 scores

        Returns:
            Dictionary with AP50 scores: {"yolov5": 0.85, "yolov8": 0.82, "yolov12": 0.71}

        Raises:
            RuntimeError: If experiment execution fails
        """
        logger.info(f"Running experiment: {self.variant_id}")

        try:
            # Step 1: Generate transformed images
            logger.info(f"  [1/4] Generating transformed images...")
            self._generate_transformed_images()
            logger.info(f"  Saved {len(self.sample_images)} sample images for logging")

            # Step 2: Run inference for all models
            logger.info(f"  [2/4] Running inference...")
            self._run_inference_all_models()

            # Step 3: Run analysis
            logger.info(f"  [3/4] Running analysis...")
            self._run_analysis()

            # Step 4: Extract results
            logger.info(f"  [4/4] Extracting results...")
            scores = self._extract_ap50_scores()

            logger.info(f"✓ Experiment complete: {self.variant_id}")
            logger.info(f"  Scores: v5={scores['yolov5']:.4f}, v8={scores['yolov8']:.4f}, v12={scores['yolov12']:.4f}")

            return scores

        except Exception as e:
            logger.error(f"✗ Experiment failed: {self.variant_id} - {e}")
            return {"yolov5": 0.0, "yolov8": 0.0, "yolov12": 0.0}

    def download_sample_images(self, local_output_dir: str):
        """
        Download sample transformed images from sandbox to local machine.

        Args:
            local_output_dir: Local directory to save images (e.g., "logs/config_samples")

        Returns:
            List of local file paths where images were saved
        """
        if not self.sample_images:
            logger.warning(f"No sample images to download for {self.variant_id}")
            return []

        local_paths = []
        local_variant_dir = Path(local_output_dir) / self.variant_id
        local_variant_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Downloading {len(self.sample_images)} sample images...")

        for sample_path in self.sample_images:
            try:
                # Get filename from path
                filename = Path(sample_path).name
                local_path = local_variant_dir / filename

                # Download from sandbox using correct API
                self.sandbox.fs.download_file(sample_path, str(local_path))
                local_paths.append(str(local_path))

            except Exception as e:
                logger.warning(f"  Failed to download {sample_path}: {e}")

        logger.info(f"  ✓ Downloaded to {local_variant_dir}")
        return local_paths

    def _generate_transformed_images(self):
        """Generate transformed images using matias_metamorphic.py."""
        images_dir = f"{self.work_dir}/data/base_images/{self.dataset}/images"
        output_dir = f"{self.work_dir}/data/transformed_images/{self.dataset}/{self.variant_id}"
        samples_dir = f"{self.work_dir}/logs/config_samples/{self.variant_id}"

        # Create output directories
        self.sandbox.process.exec(f"mkdir -p {output_dir}", timeout=5000)
        self.sandbox.process.exec(f"mkdir -p {samples_dir}", timeout=5000)

        # Get list of images
        list_result = self.sandbox.process.exec(f"ls {images_dir}", timeout=5000)
        if list_result.exit_code != 0:
            raise RuntimeError(f"Failed to list images: {list_result.result}")

        images = [img.strip() for img in list_result.result.strip().split("\n") if img.strip()]

        # Transform each image
        config_json = json.dumps(self.transform_config)

        for i, image in enumerate(images[:5]):  # Limit to 5 images for faster testing
            cmd = (
                f"cd {self.work_dir} && "
                f"python utils/matias_metamorphic.py "
                f"--rule {self.transformation} "
                f"--config '{config_json}' "
                f"--base_image {images_dir}/{image} "
                f"--output_path {output_dir}"
            )
            result = self.sandbox.process.exec(cmd, timeout=30000)
            if result.exit_code != 0:
                logger.warning(f"Transformation failed for {image}: {result.result}")

            # Save sample images for logging (first N images)
            if i < self.num_sample_images:
                # Construct transformed image filename (adds rule name suffix)
                img_name, img_ext = os.path.splitext(image)
                transformed_filename = f"{img_name}_{self.transformation}{img_ext}"
                transformed_image = f"{output_dir}/{transformed_filename}"

                sample_filename = f"sample_{i:03d}_{transformed_filename}"
                sample_path = f"{samples_dir}/{sample_filename}"

                # Copy transformed image to samples directory
                copy_result = self.sandbox.process.exec(
                    f"cp {transformed_image} {sample_path}",
                    timeout=5000
                )

                if copy_result.exit_code == 0:
                    self.sample_images.append(sample_path)
                else:
                    logger.warning(f"Failed to copy sample {i}: {copy_result.result}")

    def _run_inference_all_models(self):
        """Run inference for YOLOv5, v8, and v12."""
        images_dir = f"{self.work_dir}/data/base_images/{self.dataset}/images"
        transformed_dir = f"{self.work_dir}/data/transformed_images/{self.dataset}/{self.variant_id}"

        models = {
            "yolov5": {
                "script": "models/yolov5/infer_v5_kfold.py",
                "weights": "models/yolov5/yolov5_param_kfold.pt",
            },
            "yolov8": {
                "script": "models/yolov8/infer_v8.py",
                "weights": "models/yolov8/yolov8_param_kfold.pt",
            },
            "yolov12": {
                "script": "models/yolov12/infer_v12.py",
                "weights": "models/yolov12/yolov12_param_kfold.pt",
            },
        }

        for model_name, model_info in models.items():
            logger.info(f"    Running {model_name} inference...")

            # Infer on original images (if not already done)
            # Use bash for loop to process each image individually
            cmd_orig = (
                f"cd {self.work_dir} && "
                f"for img in {images_dir}/*; do "
                f"python {model_info['script']} "
                f"--weights {model_info['weights']} "
                f"--image \"$img\" "
                f"--output-subdir original; "
                f"done"
            )
            self.sandbox.process.exec(cmd_orig, timeout=self.timeout)

            # Infer on transformed images
            cmd_transform = (
                f"cd {self.work_dir} && "
                f"for img in {transformed_dir}/*; do "
                f"python {model_info['script']} "
                f"--weights {model_info['weights']} "
                f"--image \"$img\" "
                f"--output-subdir {self.variant_id}; "
                f"done"
            )
            result = self.sandbox.process.exec(cmd_transform, timeout=self.timeout)
            if result.exit_code != 0:
                logger.warning(f"{model_name} inference had issues: {result.result[:500]}")

    def _run_analysis(self):
        """Run analysis to compare original vs transformed detections."""
        cmd = (
            f"cd {self.work_dir} && "
            f"bash analysis/scripts/run_all_analysis_matias_meta.sh --dataset {self.dataset}"
        )
        result = self.sandbox.process.exec(cmd, timeout=self.timeout)
        if result.exit_code != 0:
            logger.warning(f"Analysis had issues: {result.result}")

    def _extract_ap50_scores(self) -> Dict[str, float]:
        """Extract AP50 scores from stats.json files."""
        scores = {}

        models = ["yolov5", "yolov8", "yolov12"]

        for model in models:
            stats_path = (
                f"{self.work_dir}/analysis/results/{model}/{self.variant_id}/"
                f"original_vs_transformed/stats.json"
            )

            try:
                # Read stats file from sandbox
                read_result = self.sandbox.process.exec(f"cat {stats_path}", timeout=5000)

                if read_result.exit_code == 0:
                    stats = json.loads(read_result.result)
                    scores[model] = float(stats.get("AP50", 0.0))
                else:
                    logger.warning(f"Could not read {stats_path}")
                    scores[model] = 0.0

            except Exception as e:
                logger.warning(f"Failed to extract {model} AP50: {e}")
                scores[model] = 0.0

        return scores


if __name__ == "__main__":
    # Test sandbox manager (requires API key)
    print("=" * 80)
    print("Sandbox Manager Test")
    print("=" * 80)

    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        print("Error: DAYTONA_API_KEY not set in environment")
        print("Set it with: export DAYTONA_API_KEY=your_key_here")
        exit(1)

    print("\n✓ API key found")
    print("Note: Actual sandbox provisioning requires valid Daytona account.")
    print("=" * 80)
