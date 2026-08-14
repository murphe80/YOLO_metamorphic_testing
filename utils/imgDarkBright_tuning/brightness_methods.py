#!/usr/bin/env python3
"""
Brightness Scoring Methods (RQ2.b)

A collection of methods for scoring the "brightness" of an image, mixing
classic colour-science approaches with a novel clustering-based approach.

Classic methods:
    - Linear Luminance (Y'): Rec.601 and Rec.709 weighted sums of RGB.
    - HSV Value (V): max(R, G, B).
    - HSI/HSL Lightness (L): (max(R, G, B) + min(R, G, B)) / 2.
    - HSI Intensity (I): (R + G + B) / 3.
    - CIELAB Lightness (L*): perceptual lightness via the CIE Lab* colour
      space (nonlinear, compensates for human visual response).
    - RMS contrast: standard deviation of the (Rec.709) luminance channel,
      capturing spread rather than average brightness.

Novel method:
    - Mean-shift separation-to-spread ratio: clusters pixel colours with
      mean-shift and scores the image by the ratio of between-cluster
      separation to within-cluster spread (see ``mean_shift.py``).

All "average brightness" methods return a score normalised to [0, 1]
(except CIELAB L*, which is returned on its natural [0, 100] scale to avoid
losing precision, and the novel mean-shift ratio, which is unbounded).

Usage:
    from utils.imgDarkBright_tuning.brightness_methods import (
        calculate_all_brightness_scores,
    )

    scores = calculate_all_brightness_scores(image_bgr)
    # {'luminance_rec601': 0.42, 'luminance_rec709': 0.40, ...}
"""

from typing import Callable, Dict

import cv2 as cv
import numpy as np

from utils.imgDarkBright_tuning.mean_shift import mean_shift_clustering_variance

REC601_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)
REC709_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _to_rgb_float(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image to an RGB float array normalised to [0, 1]."""
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    return image_rgb.astype(np.float64) / 255.0


def linear_luminance_rec601(image_bgr: np.ndarray) -> float:
    """
    Linear luminance (Y') using Rec.601 weights: 0.299R + 0.587G + 0.114B.

    Matches the weighting used by classic video luminance / analogue TV.

    Returns:
        Mean luminance in [0, 1].
    """
    rgb = _to_rgb_float(image_bgr)
    y = rgb @ REC601_WEIGHTS
    return float(np.mean(y))


def linear_luminance_rec709(image_bgr: np.ndarray) -> float:
    """
    Linear luminance (Y') using Rec.709 weights: 0.2126R + 0.7152G + 0.0722B.

    Matches the weighting used by HD video / OpenCV & scikit-image's
    default RGB-to-grayscale conversion.

    Returns:
        Mean luminance in [0, 1].
    """
    rgb = _to_rgb_float(image_bgr)
    y = rgb @ REC709_WEIGHTS
    return float(np.mean(y))


def hsv_value(image_bgr: np.ndarray) -> float:
    """
    HSV Value channel: V = max(R, G, B).

    Fast approximation of brightness; ignores perceptual nonlinearities.

    Returns:
        Mean V in [0, 1].
    """
    rgb = _to_rgb_float(image_bgr)
    v = np.max(rgb, axis=2)
    return float(np.mean(v))


def hsi_lightness(image_bgr: np.ndarray) -> float:
    """
    HSI/HSL Lightness channel: L = (max(R, G, B) + min(R, G, B)) / 2.

    Returns:
        Mean L in [0, 1].
    """
    rgb = _to_rgb_float(image_bgr)
    lightness = (np.max(rgb, axis=2) + np.min(rgb, axis=2)) / 2.0
    return float(np.mean(lightness))


def hsi_intensity(image_bgr: np.ndarray) -> float:
    """
    HSI Intensity channel: I = (R + G + B) / 3.

    Returns:
        Mean I in [0, 1].
    """
    rgb = _to_rgb_float(image_bgr)
    intensity = np.mean(rgb, axis=2)
    return float(np.mean(intensity))


def cielab_lightness(image_bgr: np.ndarray) -> float:
    """
    CIELAB Lightness (L*): perceptual lightness from the CIE Lab* colour
    space. Requires a nonlinear RGB -> XYZ -> Lab* transform (cube-root
    compression), which more closely tracks human brightness perception
    than the linear/HSx methods above.

    Returns:
        Mean L* in [0, 100] (native CIELAB scale).
    """
    rgb = _to_rgb_float(image_bgr).astype(np.float32)
    lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB)
    l_star = lab[:, :, 0]
    return float(np.mean(l_star))


def rms_contrast(image_bgr: np.ndarray) -> float:
    """
    RMS contrast: standard deviation of the Rec.709 luminance channel.

    Unlike the methods above (which measure average brightness), this
    measures the spread of brightness across the image.

    Returns:
        Standard deviation of luminance in [0, 1].
    """
    rgb = _to_rgb_float(image_bgr)
    y = rgb @ REC709_WEIGHTS
    return float(np.std(y))


def mean_shift_separation_spread_ratio(image_bgr: np.ndarray) -> float:
    """
    Novel method: separation-to-spread ratio from mean-shift clustering of
    pixel colours (see ``mean_shift.mean_shift_clustering_variance``).

    Returns:
        Ratio of weighted between-cluster separation to weighted
        within-cluster spread (unbounded, >= 0).
    """
    return float(mean_shift_clustering_variance(image_bgr))


# Registry of all brightness scoring methods, keyed by name.
BRIGHTNESS_METHODS: Dict[str, Callable[[np.ndarray], float]] = {
    "luminance_rec601": linear_luminance_rec601,
    "luminance_rec709": linear_luminance_rec709,
    "hsv_value": hsv_value,
    "hsi_lightness": hsi_lightness,
    "hsi_intensity": hsi_intensity,
    "cielab_lightness": cielab_lightness,
    "rms_contrast": rms_contrast,
    "mean_shift_separation_spread_ratio": mean_shift_separation_spread_ratio,
}


def calculate_all_brightness_scores(image_bgr: np.ndarray) -> Dict[str, float]:
    """
    Calculate every registered brightness score for a single image.

    Args:
        image_bgr: Input BGR image as loaded by cv2.imread.

    Returns:
        Dict mapping method name -> brightness score.
    """
    return {
        name: method(image_bgr)
        for name, method in BRIGHTNESS_METHODS.items()
    }
