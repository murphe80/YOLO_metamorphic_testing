# Search algorithm to find the optimal brightness transformation for the image.
# Uses golden-section search
# relationship between brightness and accuracy: both very-dark and very-bright
# images yield poor detections, with a peak somewhere in between.

import os
import json
import cv2 as cv
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matias_metamorphic import gen_metamorphic
from analysis.utils.format_detections import format_detections
from analysis.coco_eval_per_class import evaluate_coco_per_class
from models.yolov5.infer_v5_kfold import extract_detections, run_inference

BRIGHTNESS_MULTIPLIER_LOW= 0.1
BRIGHTNESS_MULTIPLIER_HIGH= 0.9
BRIGHTNESS_CONSTANT= 0
GAMMA= 1
METHOD= "scale"
THRESHOLD_AP= 0.01
MAX_ITERATIONS= 30
# COCO category IDs produced by analysis.utils.format_detections.
BLUE_CLASS_ID = 2
YELLOW_CLASS_ID = 1

RESULTS_DIR = "search_results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Golden-section ratio
_GSR = (np.sqrt(5) - 1) / 2   # ≈ 0.618


def _make_config(multiplier: float) -> dict:
    """Build a brightness config dict from a multiplier value."""
    return {
        "brightness_constant":   BRIGHTNESS_CONSTANT,
        "brightness_multiplier": float(multiplier),
        "gamma":                 GAMMA,
        "method":                METHOD,
    }


def _split_image_path(image_path: str) -> tuple[str, str]:
    """Return (stem, '.ext') from 'stem.ext'."""
    return os.path.splitext(image_path)


def get_accuracy(
    transformed_detections_filepath: str,
    gt_detections_filepath: str,
    image_stem: str,
    img_width: int,
    img_height: int,
) -> tuple[float, float]:
    """
    Format both detection files to COCO JSON and return (AP_blue, AP_yellow).
    """
    evaluation_dir = os.path.dirname(transformed_detections_filepath)
    transformed_det_path = os.path.join(evaluation_dir, "coco_transf_detections.json")
    gt_det_path          = os.path.join(evaluation_dir, "coco_gt_detections.json")

    gt_path = gt_detections_filepath
    if os.path.isdir(gt_path):
        gt_path = os.path.join(gt_path, f"{image_stem}.txt")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground-truth label not found: {gt_path}")

    format_detections(
        gt_path,
        gt_det_path,
        as_ground_truth=True,
        is_yolo_format=True,
        img_width=img_width,
        img_height=img_height,
    )
    format_detections(
        transformed_detections_filepath,
        transformed_det_path,
        gt_coco_file=gt_det_path,
    )

    # pycocotools cannot load an empty result list. No predictions means zero
    # AP for the classes being tuned.
    if os.path.getsize(transformed_detections_filepath) == 0:
        return 0.0, 0.0

    _, per_class_stats = evaluate_coco_per_class(gt_det_path, transformed_det_path)
    ap_blue   = per_class_stats.get(BLUE_CLASS_ID, {}).get("AP", float("nan"))
    ap_yellow = per_class_stats.get(YELLOW_CLASS_ID, {}).get("AP", float("nan"))
    return ap_blue, ap_yellow


def _evaluate_multiplier(
    multiplier:              float,
    image_bgr:               np.ndarray,
    image_path_tuple:        tuple[str, str],
    gt_detections_filepath:  str,
    label:                   str,
) -> tuple[str, float, float, str]:
    """
    Apply the brightness transform, run YOLOv5 inference, and return
    (output_filepath, ap_blue, ap_yellow, detections_filepath).
    `label` is used only to name the output sub-directory.
    """
    config     = _make_config(multiplier)
    output_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(output_dir, exist_ok=True)

    success, output_filepath = gen_metamorphic(
        image_bgr, "imgDarkBright", config, image_path_tuple, output_dir
    )
    if not success:
        raise RuntimeError(f"gen_metamorphic failed for label='{label}'")

    image_stem = os.path.basename(image_path_tuple[0])
    updated_detections_filepath = os.path.join(output_dir, f"{image_stem}.txt")

    print(f"[{label}] Running YOLOv5 inference (multiplier={multiplier:.4f})")
    result = run_inference("models/yolov5/yolov5_param_kfold.pt", output_filepath)
    detections = extract_detections(result)
    with open(updated_detections_filepath, "w") as f:
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            f.write(
                f'{detection["class"]} {detection["confidence"]:.6f} '
                f'{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n'
            )

    img_height, img_width = image_bgr.shape[:2]
    ap_blue, ap_yellow = get_accuracy(
        updated_detections_filepath,
        gt_detections_filepath,
        image_stem,
        img_width,
        img_height,
    )
    print(f"[{label}] AP_blue={ap_blue:.4f}  AP_yellow={ap_yellow:.4f}")
    return output_filepath, ap_blue, ap_yellow, updated_detections_filepath


def _combined_ap(ap_blue: float, ap_yellow: float) -> float:
    """
    Aggregate per-class APs into a single scalar for the search.
    Using the mean keeps both classes equally weighted.
    """
    class_aps = np.asarray([ap_blue, ap_yellow], dtype=float)
    finite_aps = class_aps[np.isfinite(class_aps)]
    return float(np.mean(finite_aps)) if finite_aps.size else 0.0

def find_peak_multiplier(
    image_bgr:              np.ndarray,
    image_path_tuple:       tuple[str, str],
    gt_detections_filepath: str,
) -> tuple[float, float, str]:
    """
    Locate the brightness multiplier that **maximises** combined AP using
    golden-section search on [BRIGHTNESS_MULTIPLIER_LOW, BRIGHTNESS_MULTIPLIER_HIGH].

    Golden-section search is used here because the accuracy curve is
    n shaped: accuracy is poor at both extremes and peaks
    somewhere in the middle.

    Returns (best_multiplier, best_combined_ap, best_output_filepath).
    """
    a = BRIGHTNESS_MULTIPLIER_LOW
    b = BRIGHTNESS_MULTIPLIER_HIGH

    # Initial interior probes
    m1 = b - _GSR * (b - a)
    m2 = a + _GSR * (b - a)

    _, ap_b1, ap_y1, _ = _evaluate_multiplier(
        multiplier=m1, image_bgr=image_bgr,
        image_path_tuple=image_path_tuple,
        gt_detections_filepath=gt_detections_filepath,
        label="peak_m1_init",
    )
    f1 = _combined_ap(ap_b1, ap_y1)

    _, ap_b2, ap_y2, _ = _evaluate_multiplier(
        multiplier=m2, image_bgr=image_bgr,
        image_path_tuple=image_path_tuple,
        gt_detections_filepath=gt_detections_filepath,
        label="peak_m2_init",
    )
    f2 = _combined_ap(ap_b2, ap_y2)

    best_filepath = ""
    iteration = 0

    while (b - a) > THRESHOLD_AP and iteration < MAX_ITERATIONS:
        iteration += 1
        if f1 < f2:
            # Peak is in [m1, b]; discard [a, m1)
            a, m1, f1 = m1, m2, f2
            m2 = a + _GSR * (b - a)
            out, ap_b2, ap_y2, _ = _evaluate_multiplier(
                multiplier=m2, image_bgr=image_bgr,
                image_path_tuple=image_path_tuple,
                gt_detections_filepath=gt_detections_filepath,
                label=f"peak_iter{iteration}_m2",
            )
            f2           = _combined_ap(ap_b2, ap_y2)
            best_filepath = out
        else:
            # Peak is in [a, m2]; discard (m2, b]
            b, m2, f2 = m2, m1, f1
            m1 = b - _GSR * (b - a)
            out, ap_b1, ap_y1, _ = _evaluate_multiplier(
                multiplier=m1, image_bgr=image_bgr,
                image_path_tuple=image_path_tuple,
                gt_detections_filepath=gt_detections_filepath,
                label=f"peak_iter{iteration}_m1",
            )
            f1            = _combined_ap(ap_b1, ap_y1)
            best_filepath = out

    best_multiplier = (a + b) / 2.0
    best_ap         = (f1 + f2) / 2.0
    return best_multiplier, best_ap, best_filepath



def search_brightness(
    image:                  np.ndarray,
    image_path:             str,
    target_accuracy:        float,
    gt_detections_filepath: str,
) -> tuple[dict, float, float, str, str]:
    """
    Find the brightness multiplier whose combined AP is closest to
    `target_accuracy`.

    Strategy
    --------
    1. Use golden-section search to find the peak multiplier (maximum AP).
    2. If peak AP < target_accuracy the target is unreachable; return the peak.
    3. Otherwise the target is crossed on both sides of the peak.  Apply
       regula-falsi independently on [low, peak] and [peak, high] to find
       both crossing points, then return the one with the higher multiplier
       (brighter image) as the preferred operating point.

    Returns (best_config, ap_blue, ap_yellow, output_filepath, updated_detections_filepath).
    """
    image_bgr        = image
    image_path_tuple = _split_image_path(image_path)

    # find peak multiplier
    peak_mult, peak_ap, peak_filepath = find_peak_multiplier(
        image_bgr, image_path_tuple, gt_detections_filepath
    )
    print(f"[search] Peak  multiplier={peak_mult:.4f}  combined_AP={peak_ap:.4f}")

    if peak_ap < target_accuracy:
        print("[search] Target accuracy is unachievable; returning peak.")
        final_path, ap_b, ap_y, det_path = _evaluate_multiplier(
            multiplier=peak_mult, image_bgr=image_bgr,
            image_path_tuple=image_path_tuple,
            gt_detections_filepath=gt_detections_filepath,
            label="final_peak",
        )
        return _make_config(peak_mult), ap_b, ap_y, final_path, det_path

    # regula-falsi on the right slope [peak_mult, HIGH]
    def _rf_search(lo_mult, hi_mult, lo_ap, hi_ap, side_label):
        """
        Regula-falsi to find multiplier where combined_AP == target_accuracy.
        Assumes lo_ap and hi_ap bracket the target (one above, one below).
        Returns (best_mult, best_ap, best_filepath).
        """
        best_mult     = lo_mult
        best_ap       = lo_ap
        best_filepath = ""

        for i in range(MAX_ITERATIONS):
            denom = hi_ap - lo_ap
            if abs(denom) < 1e-9:
                mid = (lo_mult + hi_mult) / 2.0
            else:
                # Regula falsi interpolation
                mid = lo_mult + (hi_mult - lo_mult) * (target_accuracy - lo_ap) / denom
                mid = float(np.clip(mid, min(lo_mult, hi_mult), max(lo_mult, hi_mult)))

            out, ap_b, ap_y, _ = _evaluate_multiplier(
                multiplier=mid, image_bgr=image_bgr,
                image_path_tuple=image_path_tuple,
                gt_detections_filepath=gt_detections_filepath,
                label=f"rf_{side_label}_iter{i}",
            )
            mid_ap        = _combined_ap(ap_b, ap_y)
            best_mult     = mid
            best_ap       = mid_ap
            best_filepath = out

            print(
                f"[rf/{side_label}] iter={i}  mult={mid:.4f} "
                f"AP={mid_ap:.4f}  bracket=[{lo_mult:.4f},{hi_mult:.4f}]"
            )

            if abs(mid_ap - target_accuracy) <= THRESHOLD_AP:
                break

            # Tighten bracket: mid replaces whichever endpoint is on the same
            # side of the target as mid. This is correct for both increasing
            # (left) and decreasing (right) slopes.
            if (mid_ap - target_accuracy) * (lo_ap - target_accuracy) > 0:
                lo_mult, lo_ap = mid, mid_ap
            else:
                hi_mult, hi_ap = mid, mid_ap

        return best_mult, best_ap, best_filepath

    # Evaluate endpoints of the right slope
    _, ap_b_low, ap_y_low, _ = _evaluate_multiplier(
        multiplier=BRIGHTNESS_MULTIPLIER_LOW, image_bgr=image_bgr,
        image_path_tuple=image_path_tuple,
        gt_detections_filepath=gt_detections_filepath,
        label="endpoint_low",
    )
    _, ap_b_high, ap_y_high, _ = _evaluate_multiplier(
        multiplier=BRIGHTNESS_MULTIPLIER_HIGH, image_bgr=image_bgr,
        image_path_tuple=image_path_tuple,
        gt_detections_filepath=gt_detections_filepath,
        label="endpoint_high",
    )
    ap_low  = _combined_ap(ap_b_low,  ap_y_low)
    ap_high = _combined_ap(ap_b_high, ap_y_high)

    results = []

    # Right slope: [peak_mult → HIGH]  (AP decreases)
    if ap_high < target_accuracy <= peak_ap:
        mult_r, ap_r, fp_r = _rf_search(
            peak_mult, BRIGHTNESS_MULTIPLIER_HIGH,
            peak_ap,   ap_high,
            side_label="right",
        )
        results.append((mult_r, ap_r, fp_r))
    else:
        print("[search] Target not bracketed on right slope; skipping.")

    # Left slope: [LOW → peak_mult]  (AP increases)
    if ap_low < target_accuracy <= peak_ap:
        mult_l, ap_l, fp_l = _rf_search(
            BRIGHTNESS_MULTIPLIER_LOW, peak_mult,
            ap_low,                    peak_ap,
            side_label="left",
        )
        results.append((mult_l, ap_l, fp_l))
    else:
        print("[search] Target not bracketed on left slope; skipping.")

    if not results:
        # Fallback: return the endpoint closest to the target
        if abs(ap_low - target_accuracy) < abs(ap_high - target_accuracy):
            best_mult, best_ap = BRIGHTNESS_MULTIPLIER_LOW, ap_low
        else:
            best_mult, best_ap = BRIGHTNESS_MULTIPLIER_HIGH, ap_high
        final_path, ap_b, ap_y, det_path = _evaluate_multiplier(
            multiplier=best_mult, image_bgr=image_bgr,
            image_path_tuple=image_path_tuple,
            gt_detections_filepath=gt_detections_filepath,
            label="fallback_endpoint",
        )
        return _make_config(best_mult), ap_b, ap_y, final_path, det_path

    # Prefer the solution with the lower multiplier (darker image)
    results.sort(key=lambda x: x[0])
    best_mult, _, best_fp = results[0]

    final_path, ap_b_final, ap_y_final, updated_detections_filepath = _evaluate_multiplier(
        multiplier=best_mult, image_bgr=image_bgr,
        image_path_tuple=image_path_tuple,
        gt_detections_filepath=gt_detections_filepath,
        label="final_best",
    )
    print(
        f"[search] Final  multiplier={best_mult:.4f}  "
        f"AP_blue={ap_b_final:.4f}  AP_yellow={ap_y_final:.4f}"
    )
    return _make_config(best_mult), ap_b_final, ap_y_final, final_path, updated_detections_filepath
