"""Presentation of transformation parameter settings in RQ1 plots."""


def configure_resolution_axis(axis, rule, values):
    """Show resolution decreasing from the original image (1) to zero."""
    if rule != "imgResolution":
        return
    ticks = sorted({0.0, 1.0, *values}, reverse=True)
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_xlim(1.0, 0.0)


def displayed_overlap_settings(rule, variants):
    """Select up to five evenly spaced settings, including both endpoints."""
    ordered = list(variants)
    if rule == "imgResolution":
        ordered.sort(key=lambda variant: variant["size_ratio"], reverse=True)
    if len(ordered) <= 5:
        return ordered
    return [ordered[round(index * (len(ordered) - 1) / 4)] for index in range(5)]
