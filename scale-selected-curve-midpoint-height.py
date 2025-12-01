#!/usr/bin/env python3
"""Scale bounce midpoints so the tallest bounce reaches a fixed height."""

import bpy

MAX_Z_HEIGHT = 0.5
GROUND_Z = 0.0
EPSILON = 1e-6


def is_ground(z_value: float) -> bool:
    return abs(z_value - GROUND_Z) < EPSILON


def get_control_points(spline):
    if spline.type == "BEZIER":
        return spline.bezier_points
    return spline.points


def collect_bounce_segments(points, use_cyclic_u: bool):
    segments = []
    total_points = len(points)
    if total_points < 3:
        return segments

    limit = total_points if use_cyclic_u else total_points - 2
    for start_idx in range(limit):
        mid_idx = (start_idx + 1) % total_points
        end_idx = (start_idx + 2) % total_points
        start_pt = points[start_idx]
        mid_pt = points[mid_idx]
        end_pt = points[end_idx]

        if (
            is_ground(start_pt.co.z)
            and is_ground(end_pt.co.z)
            and mid_pt.co.z > GROUND_Z + EPSILON
        ):
            distance = (end_pt.co - start_pt.co).length
            segments.append({"mid_idx": mid_idx, "distance": distance})

    return segments


def scale_midpoints(curve_obj):
    for spline in curve_obj.data.splines:
        control_points = get_control_points(spline)
        segments = collect_bounce_segments(control_points, spline.use_cyclic_u)
        if not segments:
            continue

        max_distance = max(segment["distance"] for segment in segments)
        if max_distance <= EPSILON:
            continue

        for segment in segments:
            ratio = segment["distance"] / max_distance
            control_points[segment["mid_idx"]].co.z = MAX_Z_HEIGHT * ratio


def main():
    selected_curves = [obj for obj in bpy.context.selected_objects if obj.type == "CURVE"]
    if not selected_curves:
        print("No selected curve objects found.")
        return

    for curve_obj in selected_curves:
        scale_midpoints(curve_obj)


if __name__ == "__main__":
    main()
