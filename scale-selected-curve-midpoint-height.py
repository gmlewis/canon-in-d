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
            segments.append(
                {
                    "mid_idx": mid_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "distance": distance,
                }
            )

    return segments



def main():
    selected_curves = [obj for obj in bpy.context.selected_objects if obj.type == "CURVE"]
    if not selected_curves:
        print("No selected curve objects found.")
        return

    segments = []
    for curve_obj in selected_curves:
        for spline in curve_obj.data.splines:
            control_points = get_control_points(spline)
            spline_segments = collect_bounce_segments(control_points, spline.use_cyclic_u)
            for segment in spline_segments:
                segments.append(
                    {
                        "control_points": control_points,
                        "mid_idx": segment["mid_idx"],
                        "start_idx": segment["start_idx"],
                        "end_idx": segment["end_idx"],
                        "distance": segment["distance"],
                    }
                )

    if not segments:
        print("No bounce segments found on selected curves.")
        return

    global_max = max(segment["distance"] for segment in segments)
    if global_max <= EPSILON:
        print("Maximum bounce distance is zero; nothing to scale.")
        return

    for segment in segments:
        ratio = segment["distance"] / global_max
        control_points = segment["control_points"]
        control_points[segment["mid_idx"]].co.z = MAX_Z_HEIGHT * ratio
        ground_height = MAX_Z_HEIGHT * ratio
        start_point = control_points[segment["start_idx"]]
        end_point = control_points[segment["end_idx"]]
        if hasattr(start_point, "handle_right"):
            start_point.handle_right.z = start_point.co.z + ground_height
        if hasattr(end_point, "handle_left"):
            end_point.handle_left.z = end_point.co.z + ground_height


if __name__ == "__main__":
    main()
