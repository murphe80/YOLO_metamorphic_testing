import cv2 as cv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def mean_shift_clustering_variance(
    image,
    quantile=0.1,
    sample_points=1000,
    max_points_for_fit=20000,
    eps=1e-8
):
    """
    Compute a separation-to-spread ratio using mean shift clustering.

    The score is based on:
    - between-cluster separation: weighted pairwise squared distance
      between cluster centers
    - within-cluster spread: weighted average variance inside each cluster

    Args:
        image: Input BGR image
        quantile: Bandwidth estimation quantile for mean shift
        sample_points: Number of samples for estimate_bandwidth
        max_points_for_fit: Optional cap on number of pixels used for clustering
        eps: Small constant to avoid division by zero

    Returns:
        ratio: separation to spread ratio
    """
    # Convert BGR to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Flatten image to N x 3 pixel array
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)

    # Optionally subsample for faster clustering on large images
    if len(pixels) > max_points_for_fit:
        rng = np.random.default_rng(42)
        fit_idx = rng.choice(len(pixels), size=max_points_for_fit, replace=False)
        pixels_fit = pixels[fit_idx]
    else:
        pixels_fit = pixels

    # Estimate bandwidth
    bandwidth = estimate_bandwidth(
        pixels_fit,
        quantile=quantile,
        n_samples=min(sample_points, len(pixels_fit))
    )

    # Fallback if estimate_bandwidth returns 0 or invalid
    if bandwidth is None or bandwidth <= 0:
        bandwidth = 10.0

    # Fit mean shift on subsampled or full set
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pixels_fit)

    # Predict labels for all pixels using nearest cluster center
    cluster_centers = ms.cluster_centers_.astype(np.float32)

    # Assign every pixel to nearest cluster center
    dists = np.linalg.norm(
        pixels[:, None, :] - cluster_centers[None, :, :],
        axis=2
    )
    labels = np.argmin(dists, axis=1)

    n_clusters = len(cluster_centers)

    if n_clusters == 0:
        return {
            "ratio": 0.0,
            "between_cluster_distance": 0.0,
            "within_cluster_variance": 0.0,
            "n_clusters": 0,
            "cluster_centers": np.empty((0, 3), dtype=np.float32),
            "cluster_weights": np.empty((0,), dtype=np.float32),
        }

    # Compute within-cluster variance and cluster weights
    within_cluster_variances = []
    cluster_weights = []

    for k in range(n_clusters):
        cluster_pixels = pixels[labels == k]
        weight = len(cluster_pixels) / len(pixels)
        cluster_weights.append(weight)

        if len(cluster_pixels) == 0:
            within_var = 0.0
        else:
            # Mean squared distance to cluster center
            diffs = cluster_pixels - cluster_centers[k]
            squared_dists = np.sum(diffs ** 2, axis=1)
            within_var = float(np.mean(squared_dists))

        within_cluster_variances.append(within_var)

    cluster_weights = np.array(cluster_weights, dtype=np.float32)
    within_cluster_variances = np.array(within_cluster_variances, dtype=np.float32)

    # Weighted average within-cluster variance
    within_cluster_variance = float(np.sum(cluster_weights * within_cluster_variances))

    # Weighted pairwise between-cluster squared distance
    between_cluster_distance = 0.0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            center_dist_sq = np.sum((cluster_centers[i] - cluster_centers[j]) ** 2)
            between_cluster_distance += cluster_weights[i] * cluster_weights[j] * center_dist_sq

    between_cluster_distance = float(between_cluster_distance)

    # Separation-to-spread ratio
    ratio = between_cluster_distance / (within_cluster_variance + eps)
    return ratio