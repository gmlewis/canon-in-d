#!/usr/bin/env python3
"""
Blender 5.0 script to sync the "Energy Trail" curves from pre-generated data.
Run this script from within Blender's scripting environment.

This script:
1. Reads the JSON data file and creates animation keyframes for the "Energy Trails".
"""

import bpy
import json
import sys
from mathutils.geometry import interpolate_bezier

# ============================================================================
# Configuration
# ============================================================================
# Path to the JSON file (relative to the .blend file or absolute)
JSON_FILE_PATH = "note-jumping-curves.json"

GLOBAL_FPS = 60
MUSIC_START_OFFSET_FRAMES = 120  # Frame at which music starts (2 seconds at 60 FPS)
INITIAL_ENERGY_TRAIL_X_OFFSET = -5.0
FIRST_LANDING_POINT_ENERGY_TRAIL_X_OFFSET_AT_T0 = 3.152
FRAME_TOLERANCE = 1e-4
GROUND_Z = 0.0
EPSILON = 1e-6
BEZIER_RESOLUTION = 24


def is_ground(z_value: float) -> bool:
    return abs(z_value - GROUND_Z) < EPSILON


def get_control_points(spline):
    return spline.bezier_points if spline.type == 'BEZIER' else spline.points


def collect_bounce_segments(points, use_cyclic_u: bool) -> list[dict]:
    segments = []
    point_count = len(points)
    if point_count < 3:
        return segments

    limit = point_count if use_cyclic_u else point_count - 2
    for start_idx in range(limit):
        mid_idx = (start_idx + 1) % point_count
        end_idx = (start_idx + 2) % point_count
        start_pt = points[start_idx]
        mid_pt = points[mid_idx]
        end_pt = points[end_idx]

        if (
            is_ground(start_pt.co.z)
            and is_ground(end_pt.co.z)
            and mid_pt.co.z > GROUND_Z + EPSILON
        ):
            segments.append({
                'start_idx': start_idx,
                'mid_idx': mid_idx,
                'end_idx': end_idx,
            })

    return segments


def evaluate_cubic_length(curve_obj, start_point, end_point, start_handle_attr, end_handle_attr, resolution):
    matrix = curve_obj.matrix_world
    start_co = matrix @ start_point.co
    end_co = matrix @ end_point.co
    start_handle = matrix @ getattr(start_point, start_handle_attr)
    end_handle = matrix @ getattr(end_point, end_handle_attr)
    samples = interpolate_bezier(start_co, start_handle, end_handle, end_co, resolution)
    return sum((samples[i + 1] - samples[i]).length for i in range(len(samples) - 1))


def compute_bounce_lengths(curve_obj) -> list[float]:
    lengths = []
    for spline in curve_obj.data.splines:
        control_points = get_control_points(spline)
        segments = collect_bounce_segments(control_points, spline.use_cyclic_u)
        for seg in segments:
            start_pt = control_points[seg['start_idx']]
            mid_pt = control_points[seg['mid_idx']]
            end_pt = control_points[seg['end_idx']]
            half_a = evaluate_cubic_length(
                curve_obj, start_pt, mid_pt, 'handle_right', 'handle_left', BEZIER_RESOLUTION
            )
            half_b = evaluate_cubic_length(
                curve_obj, mid_pt, end_pt, 'handle_right', 'handle_left', BEZIER_RESOLUTION
            )
            lengths.append(half_a + half_b)
    return lengths


def landing_timestamp(landing_point):
    if isinstance(landing_point, dict):
        return float(landing_point.get('timestamp', 0.0))
    if isinstance(landing_point, (list, tuple)):
        for element in landing_point:
            if isinstance(element, dict) and 'timestamp' in element:
                return float(element.get('timestamp', 0.0))
        if landing_point and isinstance(landing_point[0], (int, float)):
            return float(landing_point[0])
    return 0.0


def normalize_curve_point(point):
    if isinstance(point, dict):
        return point
    if isinstance(point, (list, tuple)):
        for candidate in point:
            if isinstance(candidate, dict):
                return candidate
    return None


def read_first_two_x_keyframes(obj):
    anim_data = obj.animation_data
    if anim_data is None or anim_data.action is None:
        return None
    for fcurve in anim_data.action.fcurves:
        if fcurve.data_path == 'location' and fcurve.array_index == 0:
            keypoints = sorted(fcurve.keyframe_points, key=lambda kp: kp.co.x)
            if len(keypoints) >= 2:
                return keypoints[0], keypoints[1]
            return None
    return None


def clear_x_keyframes(obj):
    anim_data = obj.animation_data
    if anim_data is None or anim_data.action is None:
        return
    action = anim_data.action
    x_curves = [fc for fc in action.fcurves if fc.data_path == 'location' and fc.array_index == 0]
    for fcurve in x_curves:
        action.fcurves.remove(fcurve)


def ensure_animation_data(obj):
    if obj.animation_data is None:
        obj.animation_data_create()
    if obj.animation_data.action is None:
        action = bpy.data.actions.new(name=f"{obj.name}_action")
        obj.animation_data.action = action


def load_json_data(filepath):
    """Load and parse JSON file. Returns None on failure."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"ERROR: File '{filepath}' not found.", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON file '{filepath}': {e}", file=sys.stderr)
        return None


def load_json_data_file(filepath):
    """Load the JSON structure."""
    data = load_json_data(filepath)
    if data is None:
        return None
    if 'curves' not in data:
        print(f"ERROR: JSON file '{filepath}' is missing required sections.", file=sys.stderr)
        return None
    return data


def main():
    print("=" * 70)
    print("Sync Energy Trails (Blender Script)")
    print("=" * 70)

    print(f"\nLoading JSON data from '{JSON_FILE_PATH}'...")
    json_data = load_json_data_file(JSON_FILE_PATH)
    if json_data is None:
        print("Aborting due to JSON load failure.")
        return 1

    note_jumping_curves_collection = bpy.data.collections.get('Note Jumping Curves')
    if note_jumping_curves_collection is None:
        print("ERROR: 'Note Jumping Curves' collection not found in Blender file.", file=sys.stderr)
        return 1
    curves_parent = note_jumping_curves_collection.objects['Note Jumping Curves Parent']
    if curves_parent is None:
        print("ERROR: 'Note Jumping Curves Parent' object not found in Blender file.", file=sys.stderr)
        return 1

    all_trails_collection = bpy.data.collections.get('Energy Trails')
    if all_trails_collection is None:
        print("ERROR: 'Energy Trails' collection not found in Blender file.", file=sys.stderr)
        return 1

    for trail in all_trails_collection.objects:
        if trail is None:
            print("ERROR: An Energy Trail object not found in Blender file.", file=sys.stderr)
            return 1

        trail_name = trail.name
        print(f"\nProcessing Energy Trail: '{trail_name}'")
        curve_name = trail_name.replace("Energy Trail ", "curve")
        raw_curve_entry = json_data.get('curves', {}).get(curve_name, {})
        json_curve_points = raw_curve_entry.get('points', [])
        if not json_curve_points:
            print(f"  No data found for '{curve_name}' in JSON.")
            return 1
        json_curve_landings = []
        for raw_point in json_curve_points:
            normalized = normalize_curve_point(raw_point)
            if normalized and normalized.get('type') == 'landing':
                json_curve_landings.append(normalized)
        if not json_curve_landings:
            print(f"No landing points found in '{curve_name}' in JSON.")
            return 1
        print(f"  Found {len(json_curve_landings)} landings for '{curve_name}'.")

        blender_curve = curves_parent.children.get(f"{curve_name}_curve")
        if blender_curve is None:
            print(f"ERROR: Blender curve object '{curve_name}_curve' not found.", file=sys.stderr)
            return 1

        first_two = read_first_two_x_keyframes(trail)
        if first_two is None:
            print(f"  WARNING: Unable to find at least two X keyframes for '{trail_name}'. skipping.")
            continue

        first_kf, second_kf = first_two
        if abs(first_kf.co.x - 1.0) > FRAME_TOLERANCE:
            print(
                f"  WARNING: First keyframe for '{trail_name}' is at frame {first_kf.co.x:.6f} (expected 1); skipping."
            )
            continue
        if abs(second_kf.co.x - MUSIC_START_OFFSET_FRAMES) > FRAME_TOLERANCE:
            print(
                f"  WARNING: Second keyframe for '{trail_name}' is at frame {second_kf.co.x:.6f} "
                f"(expected {MUSIC_START_OFFSET_FRAMES}); skipping."
            )
            continue

        initial_x = first_kf.co.y
        landing_x = second_kf.co.y
        print(f"  Using initial trail X {initial_x:.6f} and first landing X {landing_x:.6f} for '{trail_name}'.")

        clear_x_keyframes(trail)
        ensure_animation_data(trail)

        trail.location.x = initial_x
        trail.keyframe_insert(data_path="location", index=0, frame=1.0)
        trail.location.x = landing_x
        trail.keyframe_insert(data_path="location", index=0, frame=float(MUSIC_START_OFFSET_FRAMES))

        bounce_lengths = compute_bounce_lengths(blender_curve)
        expected_pairs = max(0, len(json_curve_landings) - 1)
        if expected_pairs == 0:
            print(f"  WARNING: Curve '{curve_name}' contains fewer than two landings; no additional keyframes generated.")
            continue
        if not bounce_lengths:
            print(f"  WARNING: No bounce segments found for '{curve_name}'; skipping.")
            continue
        if len(bounce_lengths) < expected_pairs:
            print(
                f"  WARNING: Curve '{curve_name}' had {len(bounce_lengths)} measured bounces but {expected_pairs} "
                "landing intervals; will use available segments only."
            )
        pair_count = min(expected_pairs, len(bounce_lengths))

        prev_x = landing_x
        for idx in range(pair_count):
            next_landing = json_curve_landings[idx + 1]
            segment_length = bounce_lengths[idx]
            next_x = prev_x + segment_length
            landing_frame = float(MUSIC_START_OFFSET_FRAMES) + landing_timestamp(next_landing) * GLOBAL_FPS
            trail.location.x = next_x
            trail.keyframe_insert(data_path="location", index=0, frame=landing_frame)
            prev_x = next_x

if __name__ == "__main__":
    main()
