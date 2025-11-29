#!/usr/bin/env python3
"""
Blender 5.0 script to create bezier curves from pre-generated JSON data.
Run this script from within Blender's scripting environment.

This script reads the JSON output from gen-note-jumping-curves-to-json.py
and creates all the bezier curves in Blender.

All configuration is read from the JSON file, so you can edit the JSON
to customize the curves without modifying this script.
"""

import bpy
import json
import sys
import os

# ============================================================================
# Configuration
# ============================================================================
# Path to the JSON file (relative to the .blend file or absolute)
JSON_FILE = "note-jumping-curves.json"


def load_curve_data(filepath):
    """Load the curve data JSON file."""
    # Try relative to blend file first, then current directory
    paths_to_try = [filepath]

    blend_dir = bpy.path.abspath("//")
    if blend_dir:
        paths_to_try.insert(0, os.path.join(blend_dir, filepath))

    for path in paths_to_try:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"Loaded curve data from: {path}")
            return data
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON file '{path}': {e}", file=sys.stderr)
            return None

    print(f"ERROR: Could not find JSON file '{filepath}'", file=sys.stderr)
    print(f"  Tried paths: {paths_to_try}", file=sys.stderr)
    return None


def prevent_curve_crossings(curves_data):
    """
    Ensure curves never cross in the X-Y plane by swapping Y values at noteOn events.

    At every 'landing' (noteOn) event, all curves are sorted such that their Y values
    satisfy the invariant: curve1.Y <= curve2.Y <= ... <= curveN.Y

    This function modifies the curves_data in place by swapping Y values between curves
    at each X position. No Y values are created or altered - only swapped between curves.

    Args:
        curves_data: Dict of curve data from JSON (will be modified in place)

    Returns:
        Number of X positions where swaps were performed
    """
    curve_names = sorted(curves_data.keys(), key=lambda x: int(x.replace('curve', '')))
    num_curves = len(curve_names)

    if num_curves < 2:
        return 0

    # Build an index of landing points for each curve, keyed by X position
    # For each curve, create a dict: x_position -> point_index
    curve_landing_indices = {}
    for curve_name in curve_names:
        points = curves_data[curve_name]['points']
        landing_indices = {}
        for i, point in enumerate(points):
            if point.get('type') == 'landing':
                # Use rounded X to handle floating point comparison
                x_key = round(point['x'], 6)
                landing_indices[x_key] = i
        curve_landing_indices[curve_name] = landing_indices

    # Collect all unique X positions where landing events occur (across all curves)
    all_x_positions = set()
    for curve_name in curve_names:
        all_x_positions.update(curve_landing_indices[curve_name].keys())

    # Sort X positions from lowest to highest
    sorted_x_positions = sorted(all_x_positions)

    swap_count = 0

    for x_pos in sorted_x_positions:
        # Collect all curves that have a landing at this X position
        curves_at_x = []
        for curve_name in curve_names:
            if x_pos in curve_landing_indices[curve_name]:
                point_idx = curve_landing_indices[curve_name][x_pos]
                point = curves_data[curve_name]['points'][point_idx]
                curves_at_x.append({
                    'curve_name': curve_name,
                    'point_idx': point_idx,
                    'y': point['y']
                })

        if len(curves_at_x) < 2:
            continue

        # Get the current Y values sorted ascending
        current_y_values = sorted([c['y'] for c in curves_at_x])

        # Get the curves sorted by curve number (curve1, curve2, etc.)
        curves_at_x_sorted = sorted(curves_at_x, key=lambda c: int(c['curve_name'].replace('curve', '')))

        # Check if swaps are needed: curve1 should have smallest Y, curve2 next, etc.
        needs_swap = False
        for i, curve_info in enumerate(curves_at_x_sorted):
            if abs(curve_info['y'] - current_y_values[i]) > 1e-9:
                needs_swap = True
                break

        if needs_swap:
            # Swap Y values: assign smallest Y to lowest curve number, etc.
            for i, curve_info in enumerate(curves_at_x_sorted):
                point_idx = curve_info['point_idx']
                curves_data[curve_info['curve_name']]['points'][point_idx]['y'] = current_y_values[i]
            swap_count += 1

    # Now swap Y values for peak points between the same curves
    # Peaks inherit swaps from their adjacent landings
    swap_peak_y_values(curves_data, curve_landing_indices, curve_names)

    return swap_count


def swap_peak_y_values(curves_data, curve_landing_indices, curve_names):
    """
    Swap Y values for peak points to match the swaps made at landing points.

    For each peak, we look at the previous and next landing points and determine
    what Y swaps occurred there, then apply consistent swaps to the peak.

    Args:
        curves_data: Dict of curve data (modified in place)
        curve_landing_indices: Dict mapping curve names to {x_pos: point_idx}
        curve_names: Sorted list of curve names
    """
    # Build a mapping of X position -> {curve_name: y_value} for all landings
    # This represents the FINAL (post-swap) Y values at each X position
    landing_y_by_x = {}
    for curve_name in curve_names:
        for x_pos, point_idx in curve_landing_indices[curve_name].items():
            if x_pos not in landing_y_by_x:
                landing_y_by_x[x_pos] = {}
            landing_y_by_x[x_pos][curve_name] = curves_data[curve_name]['points'][point_idx]['y']

    sorted_x_positions = sorted(landing_y_by_x.keys())

    # For each curve, process its peak points
    for curve_name in curve_names:
        points = curves_data[curve_name]['points']
        num_points = len(points)

        for i, point in enumerate(points):
            if point.get('type') != 'peak':
                continue

            peak_x = point['x']

            # Find the previous and next landing X positions for this curve
            prev_landing_x = None
            next_landing_x = None

            for x_pos in sorted_x_positions:
                if x_pos < peak_x:
                    if curve_name in landing_y_by_x.get(x_pos, {}):
                        prev_landing_x = x_pos
                elif x_pos > peak_x:
                    if curve_name in landing_y_by_x.get(x_pos, {}):
                        next_landing_x = x_pos
                        break

            if prev_landing_x is None or next_landing_x is None:
                continue

            # Get all curves that have landings at both prev and next X positions
            curves_at_prev = set(landing_y_by_x.get(prev_landing_x, {}).keys())
            curves_at_next = set(landing_y_by_x.get(next_landing_x, {}).keys())
            common_curves = curves_at_prev & curves_at_next

            if len(common_curves) < 2:
                continue

            # For each curve in common_curves, find their peak between prev and next
            # and swap Y values to maintain the same relative ordering
            peaks_to_swap = []
            for cn in common_curves:
                cn_points = curves_data[cn]['points']
                for j, p in enumerate(cn_points):
                    if p.get('type') == 'peak':
                        p_x = p['x']
                        # Check if this peak is between prev_landing_x and next_landing_x
                        if prev_landing_x < p_x < next_landing_x:
                            # Verify it's the peak between those specific landings
                            peaks_to_swap.append({
                                'curve_name': cn,
                                'point_idx': j,
                                'y': p['y']
                            })
                            break

            if len(peaks_to_swap) < 2:
                continue

            # Sort peaks by their current Y values
            current_peak_y_values = sorted([p['y'] for p in peaks_to_swap])

            # Sort curves by curve number
            peaks_to_swap_sorted = sorted(peaks_to_swap, key=lambda p: int(p['curve_name'].replace('curve', '')))

            # Assign Y values: smallest curve number gets smallest Y
            for idx, peak_info in enumerate(peaks_to_swap_sorted):
                point_idx = peak_info['point_idx']
                curves_data[peak_info['curve_name']]['points'][point_idx]['y'] = current_peak_y_values[idx]


def setup_collection(collection_name):
    """
    Set up the collection for curves.
    - If it exists, delete all objects within it.
    - If it doesn't exist, create it.
    """
    scene = bpy.context.scene
    collection = bpy.data.collections.get(collection_name)

    if collection is not None:
        print(f"Collection '{collection_name}' exists. Removing all objects...")
        objects_to_delete = list(collection.objects)
        for obj in objects_to_delete:
            bpy.data.objects.remove(obj, do_unlink=True)
        print(f"  Removed {len(objects_to_delete)} objects.")
    else:
        print(f"Creating new collection '{collection_name}'...")
        collection = bpy.data.collections.new(collection_name)
        scene.collection.children.link(collection)

    return collection


def create_parent_empty(collection, parent_name):
    """Create an empty object to serve as parent for all curves."""
    empty = bpy.data.objects.new(parent_name, None)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.empty_display_size = 1.0
    collection.objects.link(empty)
    print(f"Created parent empty '{parent_name}'")
    return empty


def create_bezier_curve(curve_name, curve_data, config, collection, parent_empty):
    """
    Create a single bezier curve from the curve data.

    Args:
        curve_name: Name of the curve (e.g., 'curve1')
        curve_data: Dict with 'points', 'landingCount', 'bezierPointCount'
        config: Configuration dict from JSON
        collection: Blender collection to add objects to
        parent_empty: Parent empty object

    Returns:
        Dict with created objects
    """
    points = curve_data['points']

    if len(points) < 2:
        print(f"  Skipping {curve_name}: not enough points")
        return None

    # Create empty parent for this curve
    curve_parent = bpy.data.objects.new(curve_name, None)
    curve_parent.empty_display_type = 'PLAIN_AXES'
    curve_parent.empty_display_size = 0.5
    collection.objects.link(curve_parent)
    curve_parent.parent = parent_empty

    # Create the bezier curve data
    curve_data_blender = bpy.data.curves.new(name=f"{curve_name}_bezier", type='CURVE')
    curve_data_blender.dimensions = '3D'
    curve_data_blender.resolution_u = config.get('curveResolution', 12)

    # Create a single spline for the entire curve
    spline = curve_data_blender.splines.new('BEZIER')

    # Add bezier points (one already exists by default)
    num_points = len(points)
    spline.bezier_points.add(num_points - 1)

    # Get Z offset for calculating handle lengths
    z_offset = config.get('maxJumpingCurveZOffset', 1.0)

    # Handle length factor - controls how "sharp" the bounce is
    # At 45 degrees, the handle length in X/Y should equal the handle length in Z
    handle_length = z_offset * 0.5  # Adjust this to control sharpness

    # Set all bezier points with custom handles for landing points
    for i, point in enumerate(points):
        bp = spline.bezier_points[i]
        x, y, z = point['x'], point['y'], point['z']
        bp.co = (x, y, z)

        point_type = point.get('type', 'unknown')

        if point_type == 'landing':
            # Landing points need sharp 45-degree angle handles
            # Use FREE handles so we can position them manually
            bp.handle_left_type = 'FREE'
            bp.handle_right_type = 'FREE'

            # Calculate handle directions based on neighboring points
            # Left handle points "up and back" (toward previous peak)
            # Right handle points "up and forward" (toward next peak)

            # Get previous and next points for direction calculation
            prev_point = points[i - 1] if i > 0 else None
            next_point = points[i + 1] if i < num_points - 1 else None

            # Left handle (entry) - coming down from the previous peak
            if prev_point:
                # Direction from previous point to this point
                dx = x - prev_point['x']
                dy = y - prev_point['y']
                # Normalize and scale
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > 0:
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                else:
                    dx_norm, dy_norm = -1, 0
                # Left handle points backward and up at 45 degrees
                bp.handle_left = (
                    x - dx_norm * handle_length,
                    y - dy_norm * handle_length,
                    z + handle_length  # Up in Z
                )
            else:
                bp.handle_left = (x - handle_length, y, z + handle_length)

            # Right handle (exit) - going up to the next peak
            if next_point:
                # Direction from this point to next point
                dx = next_point['x'] - x
                dy = next_point['y'] - y
                # Normalize and scale
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > 0:
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                else:
                    dx_norm, dy_norm = 1, 0
                # Right handle points forward and up at 45 degrees
                bp.handle_right = (
                    x + dx_norm * handle_length,
                    y + dy_norm * handle_length,
                    z + handle_length  # Up in Z
                )
            else:
                bp.handle_right = (x + handle_length, y, z + handle_length)

        elif point_type == 'peak':
            # Peak points use AUTO handles for smooth arcs
            bp.handle_left_type = 'AUTO'
            bp.handle_right_type = 'AUTO'
        elif point_type in ('fly_in', 'fly_off'):
            # Fly-in and fly-off points are hovering, use AUTO for smooth flight
            bp.handle_left_type = 'AUTO'
            bp.handle_right_type = 'AUTO'
        else:
            # Default to AUTO for unknown types
            bp.handle_left_type = 'AUTO'
            bp.handle_right_type = 'AUTO'

    # Create the curve object
    curve_obj = bpy.data.objects.new(f"{curve_name}_curve", curve_data_blender)
    collection.objects.link(curve_obj)
    curve_obj.parent = curve_parent

    return {
        'parent': curve_parent,
        'curve': curve_obj,
        'data': curve_data_blender
    }


def main():
    print("=" * 70)
    print("Create Curves from JSON (Blender Script)")
    print("=" * 70)

    # Step 1: Load JSON data
    print(f"\nLoading '{JSON_FILE}'...")
    data = load_curve_data(JSON_FILE)

    if data is None:
        print("Aborting due to JSON load failure.")
        return

    # Extract configuration and metadata
    config = data.get('config', {})
    metadata = data.get('metadata', {})
    curves_data = data.get('curves', {})

    print(f"\nMetadata from JSON:")
    print(f"  Source file: {metadata.get('sourceFile', 'unknown')}")
    print(f"  Max concurrent notes: {metadata.get('maxConcurrentNotes', 'unknown')}")
    print(f"  Song duration: {metadata.get('songStartTime', 0):.2f}s to {metadata.get('songEndTime', 0):.2f}s")
    print(f"  Total landing points: {metadata.get('totalLandingPoints', 'unknown')}")
    print(f"  Total bezier points: {metadata.get('totalBezierPoints', 'unknown')}")

    print(f"\nConfiguration:")
    print(f"  Collection name: {config.get('collectionName', 'unknown')}")
    print(f"  Parent name: {config.get('parentName', 'unknown')}")
    print(f"  Z offset: {config.get('maxJumpingCurveZOffset', 'unknown')}")
    print(f"  Curve resolution: {config.get('curveResolution', 'unknown')}")
    print(f"  Handle type: {config.get('handleType', 'unknown')}")

    # Step 2: Prevent curve crossings in X-Y plane
    print(f"\nPreventing curve crossings in X-Y plane...")
    swap_count = prevent_curve_crossings(curves_data)
    print(f"  Performed Y-value adjustments at {swap_count} X positions")

    # Step 3: Set up collection
    collection_name = config.get('collectionName', 'Note Jumping Curves')
    print(f"\nSetting up collection '{collection_name}'...")
    collection = setup_collection(collection_name)

    # Step 4: Create parent empty
    parent_name = config.get('parentName', 'Note Jumping Curves Parent')
    print(f"\nCreating parent empty...")
    parent_empty = create_parent_empty(collection, parent_name)

    # Step 5: Create all bezier curves
    print(f"\nCreating {len(curves_data)} bezier curves...")
    curve_objects = {}

    for curve_name in sorted(curves_data.keys(), key=lambda x: int(x.replace('curve', ''))):
        curve_info = curves_data[curve_name]
        result = create_bezier_curve(curve_name, curve_info, config, collection, parent_empty)
        if result:
            curve_objects[curve_name] = result
            landing_count = curve_info.get('landingCount', 0)
            bezier_count = curve_info.get('bezierPointCount', 0)
            print(f"  Created {curve_name}: {landing_count} landings, {bezier_count} bezier points")

    # Step 6: Summary
    print("\n" + "=" * 70)
    print("Curve creation complete!")
    print(f"  Collection: '{collection_name}'")
    print(f"  Parent: '{parent_name}'")
    print(f"  Curves created: {len(curve_objects)}")
    print("\nHierarchy:")
    print(f"  {collection_name} (Collection)")
    print(f"  └── {parent_name} (Empty)")
    for curve_name in sorted(curve_objects.keys(), key=lambda x: int(x.replace('curve', ''))):
        print(f"      ├── {curve_name} (Empty)")
        print(f"      │   └── {curve_name}_curve (Bezier Curve)")
    print("=" * 70)

    # Select the parent to make it easy to find
    bpy.ops.object.select_all(action='DESELECT')
    parent_empty.select_set(True)
    bpy.context.view_layer.objects.active = parent_empty
    print(f"\nSelected '{parent_name}' - use Outliner to explore the hierarchy")


if __name__ == "__main__":
    main()
