#!/usr/bin/env python3
"""
Upload Dataset to Daytona Volume

Creates a Daytona volume and uploads the fsoco_test dataset to it.
This is a one-time setup that allows all sandboxes to share the dataset.

Usage:
    python scripts/upload_dataset_to_volume.py
"""

import os
import sys
import tarfile
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from daytona_sdk import (
    Daytona,
    DaytonaConfig,
    CreateSandboxFromImageParams,
    Image,
    Resources,
    VolumeMount,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_dataset_tar():
    """Create tar archive of fsoco_test dataset."""
    logger.info("Creating dataset archive...")

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tar_path = tmp.name

    with tarfile.open(tar_path, 'w:gz') as tar:
        # Add fsoco_test dataset
        test_dir = Path("data/base_images/fsoco_test")
        if not test_dir.exists():
            test_dir = Path("data/base_images/test")

        if test_dir.exists():
            logger.info(f"  Adding {test_dir}...")
            tar.add(str(test_dir / "images"), arcname="fsoco_test/images")
            tar.add(str(test_dir / "labels"), arcname="fsoco_test/labels")
        else:
            logger.error(f"Dataset not found at {test_dir}")
            return None

        # Also add fsoco_sample for testing
        sample_dir = Path("data/base_images/fsoco_sample")
        if sample_dir.exists():
            logger.info(f"  Adding {sample_dir}...")
            tar.add(str(sample_dir / "images"), arcname="fsoco_sample/images")
            tar.add(str(sample_dir / "labels"), arcname="fsoco_sample/labels")

    size_mb = Path(tar_path).stat().st_size / (1024 * 1024)
    logger.info(f"✓ Archive created: {size_mb:.1f} MB")
    return tar_path


def upload_to_volume():
    """Create volume and upload dataset using a temporary sandbox."""
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        logger.error("DAYTONA_API_KEY not found in environment")
        return None, None

    logger.info("=" * 80)
    logger.info("Upload Dataset to Daytona Volume")
    logger.info("=" * 80)

    config = DaytonaConfig(api_key=api_key)
    daytona = Daytona(config=config)

    volume = None
    sandbox = None
    tar_path = None

    try:
        # Step 1: Create volume
        logger.info("\n[1/5] Creating volume 'yolo-fsoco-dataset'...")
        volume_name = "yolo-fsoco-dataset"

        try:
            volume = daytona.volume.get(volume_name)
            logger.info(f"✓ Volume already exists: {volume.id}")
            logger.info(f"  Name: {volume.name}")
            logger.info(f"  State: {volume.state}")
        except:
            volume = daytona.volume.create(volume_name)
            logger.info(f"✓ Volume created: {volume.id}")

        # Step 2: Create dataset archive
        logger.info("\n[2/5] Creating dataset archive...")
        tar_path = create_dataset_tar()
        if not tar_path:
            logger.error("Failed to create dataset archive")
            return None, None

        # Step 3: Provision temporary sandbox with volume mounted
        logger.info("\n[3/5] Provisioning temporary sandbox to upload data...")
        resources = Resources(cpu=2, memory=4, disk=10, gpu=0)

        volume_mount = VolumeMount(
            volume_id=volume.id,
            mount_path="/mnt/dataset"
        )

        params = CreateSandboxFromImageParams(
            image=Image.debian_slim("3.10"),
            language="python",
            resources=resources,
            volumes=[volume_mount],
        )

        sandbox = daytona.create(params=params, timeout=120)
        logger.info(f"✓ Sandbox created: {sandbox.id}")
        work_dir = sandbox.get_work_dir()

        # Step 4: Upload and extract dataset to volume
        logger.info("\n[4/5] Uploading dataset to volume...")
        sandbox.fs.upload_file(tar_path, f"{work_dir}/dataset.tar.gz")
        logger.info("  ✓ Archive uploaded to sandbox")

        logger.info("  Extracting to volume mount...")
        result = sandbox.process.exec(
            f"cd /mnt/dataset && tar -xzf {work_dir}/dataset.tar.gz",
            timeout=120000
        )

        if result.exit_code == 0:
            logger.info("  ✓ Dataset extracted to volume")
        else:
            logger.error(f"  ✗ Extraction failed: {result.result}")
            return None, None

        # Step 5: Verify dataset in volume
        logger.info("\n[5/5] Verifying dataset in volume...")
        result = sandbox.process.exec("ls -lh /mnt/dataset/", timeout=10000)
        logger.info(f"  Volume contents:\n{result.result}")

        result = sandbox.process.exec("find /mnt/dataset -type f | wc -l", timeout=10000)
        logger.info(f"  Total files in volume: {result.result.strip()}")

        logger.info("\n" + "=" * 80)
        logger.info("Upload Complete!")
        logger.info("=" * 80)
        logger.info(f"\nVolume ID: {volume.id}")
        logger.info(f"Volume Name: {volume.name}")
        logger.info(f"\nDataset is now available at /mnt/dataset in all sandboxes")
        logger.info("that mount this volume.")

        # Save volume ID to file for easy reference
        volume_info_path = Path(".daytona_volume_id")
        volume_info_path.write_text(f"{volume.id}\n{volume.name}\n")
        logger.info(f"\n✓ Volume ID saved to {volume_info_path}")

        return volume.id, volume.name

    except Exception as e:
        logger.error(f"\n✗ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    finally:
        # Cleanup
        if tar_path and Path(tar_path).exists():
            Path(tar_path).unlink()
            logger.info(f"\nCleaned up temporary archive")

        if sandbox:
            logger.info(f"\n[CLEANUP] Deleting temporary sandbox {sandbox.id}...")
            try:
                sandbox.delete(timeout=60)
                logger.info("✓ Sandbox deleted")
            except Exception as e:
                logger.warning(f"Warning: Failed to delete sandbox: {e}")


if __name__ == "__main__":
    volume_id, volume_name = upload_to_volume()
    if volume_id:
        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS - Dataset uploaded to volume")
        logger.info("=" * 80)
        logger.info(f"\nUse this volume_id in your experiments:")
        logger.info(f"  {volume_id}")
        sys.exit(0)
    else:
        logger.error("\nFailed to upload dataset to volume")
        sys.exit(1)
