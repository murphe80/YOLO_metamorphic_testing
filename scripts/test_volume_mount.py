#!/usr/bin/env python3
"""
Test Volume Mount - Verify sandboxes can access dataset from volume

Usage:
    python scripts/test_volume_mount.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from daytona_sdk import Daytona, DaytonaConfig
from utils.sandbox_manager import SandboxManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_volume_mount():
    """Test that sandboxes can access dataset from mounted volume."""
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        logger.error("DAYTONA_API_KEY not found")
        return False

    # Read volume ID
    volume_id_file = Path(".daytona_volume_id")
    if not volume_id_file.exists():
        logger.error(".daytona_volume_id file not found")
        logger.error("Run: python scripts/upload_dataset_to_volume.py")
        return False

    volume_id = volume_id_file.read_text().strip().split("\n")[0]

    logger.info("=" * 80)
    logger.info("Testing Volume Mount Access")
    logger.info("=" * 80)
    logger.info(f"Volume ID: {volume_id}")

    try:
        # Create sandbox manager with volume
        manager = SandboxManager(
            api_key=api_key,
            num_sandboxes=1,
            cpu=4,
            memory=8,
            disk=10,
            gpu_count=0,
            dataset_volume_id=volume_id,
        )

        # Provision and setup
        logger.info("\n[1/3] Provisioning sandbox with volume...")
        manager.provision_sandboxes()
        sandbox = manager.sandboxes[0]

        logger.info("\n[2/3] Setting up sandbox (includes symlink creation)...")
        manager.setup_sandbox(sandbox)

        # Test dataset access
        logger.info("\n[3/3] Testing dataset access...")
        work_dir = sandbox.get_work_dir()

        # Check volume mount
        logger.info("  Checking volume mount at /mnt/dataset...")
        result = sandbox.process.exec("ls -lh /mnt/dataset/", timeout=10000)
        logger.info(f"  Volume contents:\n{result.result}")

        # Check symlink
        logger.info("  Checking symlink at data/base_images...")
        result = sandbox.process.exec(f"ls -lh {work_dir}/data/base_images/", timeout=10000)
        logger.info(f"  Symlink target:\n{result.result}")

        # Check fsoco_sample dataset
        logger.info("  Checking fsoco_sample images...")
        result = sandbox.process.exec(
            f"ls {work_dir}/data/base_images/fsoco_sample/images/ | head -5",
            timeout=10000
        )
        if result.exit_code == 0:
            logger.info(f"  Sample images:\n{result.result}")

            # Count total images
            result = sandbox.process.exec(
                f"ls {work_dir}/data/base_images/fsoco_sample/images/ | wc -l",
                timeout=10000
            )
            logger.info(f"  Total fsoco_sample images: {result.result.strip()}")
        else:
            logger.error(f"  ✗ Failed to access images: {result.result}")
            return False

        # Check fsoco_test dataset
        logger.info("  Checking fsoco_test images...")
        result = sandbox.process.exec(
            f"ls {work_dir}/data/base_images/fsoco_test/images/ | wc -l",
            timeout=10000
        )
        if result.exit_code == 0:
            logger.info(f"  Total fsoco_test images: {result.result.strip()}")
        else:
            logger.error(f"  ✗ Failed to access fsoco_test: {result.result}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ SUCCESS - Dataset accessible from volume!")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if manager and manager.sandboxes:
            logger.info("\n[CLEANUP] Deleting sandbox...")
            manager.cleanup_sandboxes()


if __name__ == "__main__":
    success = test_volume_mount()
    sys.exit(0 if success else 1)
