"""
Helper functions for Daytona volume management.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_dataset_volume_id() -> str:
    """
    Get the dataset volume ID from the .daytona_volume_id file.

    Returns:
        str: Volume ID

    Raises:
        FileNotFoundError: If .daytona_volume_id file doesn't exist
        RuntimeError: If volume ID cannot be read
    """
    volume_file = Path(".daytona_volume_id")

    if not volume_file.exists():
        raise FileNotFoundError(
            "Dataset volume not found. Run: python scripts/upload_dataset_to_volume.py"
        )

    try:
        content = volume_file.read_text().strip().split("\n")
        volume_id = content[0].strip()

        if not volume_id:
            raise RuntimeError("Volume ID is empty in .daytona_volume_id")

        logger.info(f"Loaded dataset volume ID: {volume_id[:12]}...")
        return volume_id

    except Exception as e:
        raise RuntimeError(f"Failed to read volume ID from {volume_file}: {e}")


def save_dataset_volume_id(volume_id: str, volume_name: str):
    """
    Save the dataset volume ID to .daytona_volume_id file.

    Args:
        volume_id: Daytona volume ID
        volume_name: Daytona volume name
    """
    volume_file = Path(".daytona_volume_id")
    volume_file.write_text(f"{volume_id}\n{volume_name}\n")
    logger.info(f"Saved volume ID to {volume_file}")
