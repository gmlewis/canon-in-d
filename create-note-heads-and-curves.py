#!/usr/bin/env python3
"""
Blender 5.0 script to create bezier curves and note heads from pre-generated data.
Run this script from within Blender's scripting environment.

This script:
1. Reads the JSON output from gen-note-jumping-curves-to-json.py
   and creates all the bezier curves in Blender.
2. Imports note head SVG elements, scales them, and organizes them
   in a dedicated collection.

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

# Note Heads SVG file
NOTE_HEADS_SVG_FILE = "Canon_in_D-single-svg-printing_NoteHeads_renamed.svg"
NOTE_HEADS_COLLECTION_NAME = "Note Heads"
NOTE_HEADS_PARENT_NAME = "Note Heads Parent"
NOTE_HEADS_SCALE = 100.0


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

    # Recalculate Y values for all peak points based on their adjacent landings
    # This ensures peaks remain at the exact midpoint after landing Y swaps
    recalculate_peak_y_values(curves_data, curve_names)

    return swap_count


def recalculate_peak_y_values(curves_data, curve_names):
    """
    Recalculate X and Y values for all arc_peak points based on their adjacent landings.

    After landing positions have been swapped to prevent crossings, the peak X and Y values
    must be recalculated as the exact midpoint of the previous and next landing positions.
    This ensures peaks remain at the geometric midpoint for orthographic top-view alignment.

    NOTE: Z values are NOT changed - they represent the arc height and should stay fixed.

    Args:
        curves_data: Dict of curve data (modified in place)
        curve_names: Sorted list of curve names
    """
    for curve_name in curve_names:
        points = curves_data[curve_name]['points']
        num_points = len(points)

        for i, point in enumerate(points):
            # Only process arc_peak points (not fly_in_peak or fly_off_peak)
            if point.get('pointType') != 'arc_peak':
                continue

            # Find the previous landing point
            prev_landing = None
            for j in range(i - 1, -1, -1):
                if points[j].get('type') == 'landing':
                    prev_landing = points[j]
                    break

            # Find the next landing point
            next_landing = None
            for j in range(i + 1, num_points):
                if points[j].get('type') == 'landing':
                    next_landing = points[j]
                    break

            if prev_landing is None or next_landing is None:
                continue

            # Recalculate X as the exact midpoint of the two landings
            new_x = (prev_landing['x'] + next_landing['x']) / 2.0
            point['x'] = new_x

            # Recalculate Y as the exact midpoint of the two landings
            new_y = (prev_landing['y'] + next_landing['y']) / 2.0
            point['y'] = new_y

            # Also update svgX and svgY to match
            if 'svgX' in prev_landing and 'svgX' in next_landing:
                point['svgX'] = (prev_landing['svgX'] + next_landing['svgX']) / 2.0
            if 'svgY' in prev_landing and 'svgY' in next_landing:
                point['svgY'] = (prev_landing['svgY'] + next_landing['svgY']) / 2.0


def optimize_path_grouping(curves_data):
    """
    Optimize curve path assignments to maintain the invariant:
    curve1.Y <= curve2.Y <= ... <= curveN.Y at ALL timestamps.

    This invariant MUST be true at every point in time to prevent curve crossings.

    ALGORITHM:
    When we detect that curve_i will end up with Y > curve_j.Y (where i < j) at some
    future timestamp, we need to swap their ENTIRE paths from the current divergence
    point forward. This ensures the lower-numbered curve stays below the higher-numbered one.

    The key insight is that we must swap complete path segments, not individual landings,
    because curves may have different numbers of landings between timestamps.

    Args:
        curves_data: Dict of curve data from JSON (will be modified in place)

    Returns:
        Number of swaps performed
    """
    curve_names = sorted(curves_data.keys(), key=lambda x: int(x.replace('curve', '')))
    num_curves = len(curve_names)

    if num_curves < 2:
        return 0

    def get_curve_landings(curves_data, curve_name):
        """Get list of (timestamp, y, point_index) for all landings of a curve."""
        points = curves_data[curve_name]['points']
        landings = []
        for i, pt in enumerate(points):
            if pt.get('type') == 'landing':
                landings.append((pt.get('timestamp', 0), pt['y'], i))
        return sorted(landings, key=lambda x: x[0])

    def interpolate_y_at_timestamp(curve_landings, t):
        """Get the Y value of a curve at timestamp t (interpolated if between landings)."""
        if not curve_landings:
            return None
        if t <= curve_landings[0][0]:
            return curve_landings[0][1]
        if t >= curve_landings[-1][0]:
            return curve_landings[-1][1]
        for i in range(len(curve_landings) - 1):
            t1, y1, _ = curve_landings[i]
            t2, y2, _ = curve_landings[i + 1]
            if t1 <= t <= t2:
                if t2 == t1:
                    return y1
                ratio = (t - t1) / (t2 - t1)
                return y1 + ratio * (y2 - y1)
        return curve_landings[-1][1]

    def find_last_common_timestamp(curve_a_landings, curve_b_landings, before_t):
        """Find the last timestamp before before_t where both curves have landings."""
        a_times = {round(t, 6) for t, _, _ in curve_a_landings if t < before_t}
        b_times = {round(t, 6) for t, _, _ in curve_b_landings if t < before_t}
        common = a_times & b_times
        if common:
            return max(common)
        return None

    def swap_curve_segment(curves_data, curve_a, curve_b, from_t, to_t):
        """
        Swap all landing data between curve_a and curve_b from from_t to to_t (inclusive).
        Returns number of swaps made.
        """
        points_a = curves_data[curve_a]['points']
        points_b = curves_data[curve_b]['points']

        # Find all landing indices in the time range for each curve
        a_indices = []
        b_indices = []

        for i, pt in enumerate(points_a):
            if pt.get('type') == 'landing':
                t = pt.get('timestamp', 0)
                if from_t <= t <= to_t:
                    a_indices.append(i)

        for i, pt in enumerate(points_b):
            if pt.get('type') == 'landing':
                t = pt.get('timestamp', 0)
                if from_t <= t <= to_t:
                    b_indices.append(i)

        # We can only swap if both have the same number of landings in range
        # (This is the common case when curves are synchronized)
        if len(a_indices) != len(b_indices) or len(a_indices) == 0:
            return 0

        # Swap the landing data
        swap_keys = ['x', 'y', 'svgX', 'svgY', 'note', 'noteName']
        for idx_a, idx_b in zip(a_indices, b_indices):
            for key in swap_keys:
                if key in points_a[idx_a] and key in points_b[idx_b]:
                    points_a[idx_a][key], points_b[idx_b][key] = \
                        points_b[idx_b][key], points_a[idx_a][key]

        return len(a_indices)

    # Get all unique timestamps
    all_curve_landings = {cn: get_curve_landings(curves_data, cn) for cn in curve_names}
    all_timestamps = set()
    for cn in curve_names:
        for t, _, _ in all_curve_landings[cn]:
            all_timestamps.add(round(t, 6))
    sorted_timestamps = sorted(all_timestamps)

    if len(sorted_timestamps) < 2:
        return 0

    swap_count = 0
    max_passes = 100

    for pass_num in range(max_passes):
        swaps_this_pass = 0
        all_curve_landings = {cn: get_curve_landings(curves_data, cn) for cn in curve_names}

        # Check each consecutive pair of timestamps for invariant violations
        for t_idx in range(len(sorted_timestamps) - 1):
            t_current = sorted_timestamps[t_idx]
            t_next = sorted_timestamps[t_idx + 1]

            # Get Y values at both timestamps for all curves
            y_current = {cn: interpolate_y_at_timestamp(all_curve_landings[cn], t_current) for cn in curve_names}
            y_next = {cn: interpolate_y_at_timestamp(all_curve_landings[cn], t_next) for cn in curve_names}

            # Check invariant at t_next
            for i in range(len(curve_names) - 1):
                curve_lo = curve_names[i]      # Lower curve number (should have lower or equal Y)
                curve_hi = curve_names[i + 1]  # Higher curve number (should have higher or equal Y)

                if y_next[curve_lo] is None or y_next[curve_hi] is None:
                    continue

                # Invariant: curve_lo.Y <= curve_hi.Y
                TOLERANCE = 0.001
                if y_next[curve_lo] > y_next[curve_hi] + TOLERANCE:
                    # Violation! curve_lo has higher Y than curve_hi at t_next
                    # Need to swap their paths

                    # Find where to start swapping - the last common landing point
                    common_t = find_last_common_timestamp(
                        all_curve_landings[curve_lo],
                        all_curve_landings[curve_hi],
                        t_next
                    )

                    if common_t is not None:
                        # Swap from just after common_t to the end of the piece
                        end_t = max(sorted_timestamps) + 1
                        swaps = swap_curve_segment(curves_data, curve_lo, curve_hi, common_t + 0.001, end_t)
                        if swaps > 0:
                            swaps_this_pass += swaps
                            # Rebuild landing data
                            all_curve_landings = {cn: get_curve_landings(curves_data, cn) for cn in curve_names}
                    else:
                        # No common point found, try swapping just the violating landing
                        swaps = swap_curve_segment(curves_data, curve_lo, curve_hi, t_next - 0.001, t_next + 0.001)
                        if swaps > 0:
                            swaps_this_pass += swaps
                            all_curve_landings = {cn: get_curve_landings(curves_data, cn) for cn in curve_names}

        swap_count += swaps_this_pass
        if swaps_this_pass == 0:
            break

    # Recalculate peak positions after all swaps
    recalculate_peak_y_values(curves_data, curve_names)

    return swap_count


def find_divergence_point(curves_data, curve_a, curve_b, current_t, prior_timestamps, curve_landing_indices):
    """
    Find the timestamp where curve_a and curve_b diverged from a common path.

    We work backwards through prior_timestamps to find where both curves
    were at the same Y position (landing on the same note). This is the point
    from which we should swap their paths.

    Args:
        curves_data: Dict of curve data
        curve_a, curve_b: Names of the two curves that are crossing
        current_t: The timestamp where we detected the crossing
        prior_timestamps: List of timestamps before and including current_t
        curve_landing_indices: Mapping of curve_name -> timestamp -> point_index

    Returns:
        The timestamp where divergence occurred, or None if not found
    """
    Y_TOLERANCE = 0.01  # How close Y values need to be to be considered "same position"

    # Work backwards through prior timestamps
    for t_pos in reversed(prior_timestamps):
        # Check if both curves have landings at this timestamp
        if t_pos not in curve_landing_indices[curve_a] or \
           t_pos not in curve_landing_indices[curve_b]:
            continue

        idx_a = curve_landing_indices[curve_a][t_pos]
        idx_b = curve_landing_indices[curve_b][t_pos]
        y_a = curves_data[curve_a]['points'][idx_a]['y']
        y_b = curves_data[curve_b]['points'][idx_b]['y']

        # If they're at the same Y position, this is where they were together
        if abs(y_a - y_b) < Y_TOLERANCE:
            return t_pos

        # Also check if they're on the same MIDI note
        note_a = curves_data[curve_a]['points'][idx_a].get('note', -1)
        note_b = curves_data[curve_b]['points'][idx_b].get('note', -1)
        if note_a == note_b and note_a != -1:
            return t_pos

    # If we didn't find a clear divergence point, return the earliest timestamp
    # where both curves have landings
    for t_pos in prior_timestamps:
        if t_pos in curve_landing_indices[curve_a] and \
           t_pos in curve_landing_indices[curve_b]:
            return t_pos

    return None


def swap_curve_paths(curves_data, curve_a, curve_b, start_t, end_t, sorted_timestamps, curve_landing_indices):
    """
    Swap the paths of curve_a and curve_b from start_t to end_t (inclusive).

    This swaps all point data (x, y, svgX, svgY, note, noteName) between
    the two curves for all landing points in the specified range.
    Also swaps the associated peak points.

    Args:
        curves_data: Dict of curve data (modified in place)
        curve_a, curve_b: Names of the two curves to swap
        start_t: Timestamp to start swapping (inclusive)
        end_t: Timestamp to end swapping (inclusive)
        sorted_timestamps: All timestamps in chronological order
        curve_landing_indices: Mapping of curve_name -> timestamp -> point_index

    Returns:
        True if swap was performed, False otherwise
    """
    # Find the timestamps in the swap range
    swap_timestamps = [t for t in sorted_timestamps if start_t <= t <= end_t]

    if not swap_timestamps:
        return False

    points_a = curves_data[curve_a]['points']
    points_b = curves_data[curve_b]['points']

    # For each timestamp in the range, swap the landing point data
    for t_pos in swap_timestamps:
        if t_pos not in curve_landing_indices[curve_a] or \
           t_pos not in curve_landing_indices[curve_b]:
            continue

        idx_a = curve_landing_indices[curve_a][t_pos]
        idx_b = curve_landing_indices[curve_b][t_pos]

        # Swap the key landing point attributes (including x and svgX for proper positioning)
        for key in ['x', 'y', 'svgX', 'svgY', 'note', 'noteName']:
            if key in points_a[idx_a] and key in points_b[idx_b]:
                points_a[idx_a][key], points_b[idx_b][key] = \
                    points_b[idx_b][key], points_a[idx_a][key]

        # Also swap the arc_peak point that precedes this landing (if exists)
        # The arc_peak is typically at idx - 1
        if idx_a > 0 and points_a[idx_a - 1].get('pointType') == 'arc_peak' and \
           idx_b > 0 and points_b[idx_b - 1].get('pointType') == 'arc_peak':
            peak_idx_a = idx_a - 1
            peak_idx_b = idx_b - 1
            for key in ['y', 'svgY', 'noteName']:
                if key in points_a[peak_idx_a] and key in points_b[peak_idx_b]:
                    points_a[peak_idx_a][key], points_b[peak_idx_b][key] = \
                        points_b[peak_idx_b][key], points_a[peak_idx_a][key]

    return True


def set_up_collection(collection_name):
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


def align_fly_out_segment(points):
    """Ensure fly-out points reuse the Y coordinate from the final landing."""
    last_landing = None
    for point in reversed(points):
        if point.get('type') == 'landing':
            last_landing = point
            break

    if not last_landing:
        return

    last_y = last_landing.get('y')
    last_svg_y = last_landing.get('svgY')

    for point in reversed(points):
        if point.get('pointType') in ('fly_off_peak', 'fly_off_end') or point.get('type') == 'fly_off':
            if last_y is not None:
                point['y'] = last_y
            if last_svg_y is not None:
                point['svgY'] = last_svg_y
        if point is last_landing:
            break


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
    align_fly_out_segment(points)

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


# ============================================================================
# Note Heads Functions
# ============================================================================

def set_up_note_heads_collection(collection_name):
    """
    Set up the collection for note heads.
    - If it exists, delete ALL children within it (recursively).
    - If it doesn't exist, create it.
    """
    scene = bpy.context.scene
    collection = bpy.data.collections.get(collection_name)

    if collection is not None:
        print(f"Collection '{collection_name}' exists. Removing all children...")
        # Delete all objects in the collection recursively
        objects_to_delete = []

        def collect_objects(coll):
            for obj in coll.objects:
                objects_to_delete.append(obj)
            for child_coll in coll.children:
                collect_objects(child_coll)

        collect_objects(collection)

        for obj in objects_to_delete:
            bpy.data.objects.remove(obj, do_unlink=True)

        # Also remove any child collections
        child_collections = list(collection.children)
        for child_coll in child_collections:
            bpy.data.collections.remove(child_coll)

        print(f"  Removed {len(objects_to_delete)} objects.")
    else:
        print(f"Creating new collection '{collection_name}'...")
        collection = bpy.data.collections.new(collection_name)
        scene.collection.children.link(collection)

    return collection


def create_note_heads_parent(collection, parent_name):
    """Create an empty axis object to serve as parent for all note heads."""
    empty = bpy.data.objects.new(parent_name, None)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.empty_display_size = 1.0
    collection.objects.link(empty)
    print(f"Created note heads parent empty '{parent_name}'")
    return empty


def find_svg_file(filename):
    """Find the SVG file, trying multiple paths."""
    paths_to_try = [filename]

    blend_dir = bpy.path.abspath("//")
    if blend_dir:
        paths_to_try.insert(0, os.path.join(blend_dir, filename))

    for path in paths_to_try:
        if os.path.exists(path):
            return path

    return None


def import_and_setup_note_heads(collection, parent_empty, svg_filename, scale_factor):
    """
    Import SVG note heads, scale them, apply scale, and parent to the empty.

    Args:
        collection: Blender collection to add objects to
        parent_empty: Parent empty object for note heads
        svg_filename: Name of the SVG file to import
        scale_factor: Scale factor to apply (e.g., 100.0)

    Returns:
        List of imported objects
    """
    # Find the SVG file
    svg_path = find_svg_file(svg_filename)
    if svg_path is None:
        print(f"ERROR: Could not find SVG file '{svg_filename}'", file=sys.stderr)
        return []

    print(f"Importing SVG from: {svg_path}")

    # Store existing objects to identify newly imported ones
    existing_objects = set(bpy.data.objects[:])

    # Import the SVG
    bpy.ops.import_curve.svg(filepath=svg_path)

    # Find newly imported objects
    new_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]
    print(f"  Imported {len(new_objects)} objects from SVG")

    if not new_objects:
        print("  WARNING: No objects were imported from the SVG")
        return []

    # Deselect all, then select only the new objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in new_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = new_objects[0]

    # Scale all imported objects
    print(f"  Scaling all imported objects by {scale_factor}...")
    for obj in new_objects:
        obj.scale = (scale_factor, scale_factor, scale_factor)

    # Apply scale to all imported objects
    print(f"  Applying scale to all imported objects...")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Move objects to the Note Heads collection and parent them
    print(f"  Parenting objects to '{parent_empty.name}'...")
    for obj in new_objects:
        # Remove from any existing collections
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        # Add to our collection
        collection.objects.link(obj)
        # Parent to the empty
        obj.parent = parent_empty

    print(f"  Successfully set up {len(new_objects)} note head objects")
    return new_objects


def set_up_note_heads():
    """
    Main function to set up note heads:
    1. Set up the Note Heads collection (delete existing children or create new)
    2. Create a Note Heads Parent empty
    3. Import SVG, scale, apply scale, and parent all elements
    """
    print("\n" + "=" * 70)
    print("Setting up Note Heads")
    print("=" * 70)

    # Step 1: Set up the collection
    print(f"\nSetting up collection '{NOTE_HEADS_COLLECTION_NAME}'...")
    collection = set_up_note_heads_collection(NOTE_HEADS_COLLECTION_NAME)

    # Step 2: Create parent empty
    print(f"\nCreating parent empty...")
    parent_empty = create_note_heads_parent(collection, NOTE_HEADS_PARENT_NAME)

    # Step 3: Import and set up note heads
    print(f"\nImporting note heads from '{NOTE_HEADS_SVG_FILE}'...")
    note_head_objects = import_and_setup_note_heads(
        collection, parent_empty, NOTE_HEADS_SVG_FILE, NOTE_HEADS_SCALE
    )

    # Summary
    print("\n" + "-" * 70)
    print("Note Heads setup complete!")
    print(f"  Collection: '{NOTE_HEADS_COLLECTION_NAME}'")
    print(f"  Parent: '{NOTE_HEADS_PARENT_NAME}'")
    print(f"  Note head objects: {len(note_head_objects)}")
    print("-" * 70)

    return {
        'collection': collection,
        'parent': parent_empty,
        'objects': note_head_objects
    }


def main():
    print("=" * 70)
    print("Create Note Heads and Curves (Blender Script)")
    print("=" * 70)

    # ========================================================================
    # Part 1: Set up Note Heads
    # ========================================================================
    note_heads_result = set_up_note_heads()

    # ========================================================================
    # Part 2: Create Bezier Curves from JSON
    # ========================================================================
    print("\n" + "=" * 70)
    print("Creating Bezier Curves from JSON")
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

    # Step 2b: Optimize path grouping to prevent visual crossovers
    print(f"\nOptimizing path grouping to prevent visual crossovers...")
    path_swap_count = optimize_path_grouping(curves_data)
    print(f"  Performed {path_swap_count} path swaps to fix crossovers")

    # Step 3: Set up collection
    collection_name = config.get('collectionName', 'Note Jumping Curves')
    print(f"\nSetting up collection '{collection_name}'...")
    collection = set_up_collection(collection_name)

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
    print("Script execution complete!")
    print("=" * 70)

    print("\nNote Heads:")
    print(f"  Collection: '{NOTE_HEADS_COLLECTION_NAME}'")
    print(f"  Parent: '{NOTE_HEADS_PARENT_NAME}'")
    print(f"  Objects: {len(note_heads_result.get('objects', []))}")

    print(f"\nBezier Curves:")
    print(f"  Collection: '{collection_name}'")
    print(f"  Parent: '{parent_name}'")
    print(f"  Curves created: {len(curve_objects)}")
    print("\nCurve Hierarchy:")
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
