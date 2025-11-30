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
    Optimize curve path assignments to prevent X-Y plane crossovers.

    PROBLEM:
    Curves are initially assigned by pitch order at each timestamp. This can cause
    visual crossings when a curve that was in the bass clef suddenly jumps to a
    high treble note, crossing over curves that are in between.

    EXAMPLE (from the user's description):
    - note C#4 at t=18.9s has 3 curves bouncing from it
    - One curve correctly goes to D3 (bass clef) - this is fine
    - Two curves incorrectly go to D5 and F#5 (treble clef), but they CROSS over
      other curves to get there. They should have come from C#5 instead.

    SOLUTION:
    Work backwards from detected crossings to find where curves should have diverged
    earlier to maintain non-crossing paths. When a crossing is detected, trace back
    to find a point where the curves could have been swapped to avoid the crossing.

    ALGORITHM:
    1. Build a timeline of all landing events across all curves
    2. At each pair of consecutive timestamps, check if any curves cross
       (their Y order swaps between the two timestamps)
    3. When a crossing is detected, work backwards to find a "divergence point" where
       the crossing curves were at the same or similar Y position (on same note)
    4. Swap the curve assignments from that divergence point forward to the crossing point

    The key insight is: if curve A is lower than curve B at time T1, but higher at T2,
    they must have crossed. We trace back to find when they were together and swap
    their assignments from that point.

    Args:
        curves_data: Dict of curve data from JSON (will be modified in place)

    Returns:
        Number of path swaps performed
    """
    curve_names = sorted(curves_data.keys(), key=lambda x: int(x.replace('curve', '')))
    num_curves = len(curve_names)

    if num_curves < 2:
        return 0

    # Build index: for each curve, map x_position (rounded) -> point_index for landings
    def build_landing_index(curves_data, curve_names):
        curve_landing_indices = {}
        for curve_name in curve_names:
            points = curves_data[curve_name]['points']
            landing_indices = {}
            for i, point in enumerate(points):
                if point.get('type') == 'landing':
                    x_key = round(point['x'], 6)
                    landing_indices[x_key] = i
            curve_landing_indices[curve_name] = landing_indices
        return curve_landing_indices

    curve_landing_indices = build_landing_index(curves_data, curve_names)

    # Get all unique X positions where landings occur, sorted chronologically
    all_x_positions = set()
    for curve_name in curve_names:
        all_x_positions.update(curve_landing_indices[curve_name].keys())
    sorted_x_positions = sorted(all_x_positions)

    if len(sorted_x_positions) < 2:
        return 0

    swap_count = 0

    # Do multiple passes since fixing one crossing might reveal or fix others
    max_passes = 20
    for pass_num in range(max_passes):
        crossings_fixed_this_pass = 0

        # Rebuild the landing index after any swaps from previous pass
        curve_landing_indices = build_landing_index(curves_data, curve_names)

        # Check each pair of consecutive X positions for crossings
        for x_idx in range(len(sorted_x_positions) - 1):
            x_current = sorted_x_positions[x_idx]
            x_next = sorted_x_positions[x_idx + 1]

            # Get all curves that have landings at BOTH positions
            curves_at_both = []
            for curve_name in curve_names:
                if x_current in curve_landing_indices[curve_name] and \
                   x_next in curve_landing_indices[curve_name]:
                    curves_at_both.append(curve_name)

            if len(curves_at_both) < 2:
                continue

            # Get Y values at current and next position for these curves
            y_current = {}
            y_next = {}
            note_current = {}
            note_next = {}
            for curve_name in curves_at_both:
                idx_current = curve_landing_indices[curve_name][x_current]
                idx_next = curve_landing_indices[curve_name][x_next]
                y_current[curve_name] = curves_data[curve_name]['points'][idx_current]['y']
                y_next[curve_name] = curves_data[curve_name]['points'][idx_next]['y']
                note_current[curve_name] = curves_data[curve_name]['points'][idx_current].get('note', 0)
                note_next[curve_name] = curves_data[curve_name]['points'][idx_next].get('note', 0)

            # Check for crossings: curves whose relative Y order changes
            for i, curve_a in enumerate(curves_at_both):
                for curve_b in curves_at_both[i+1:]:
                    # Compare relative positions (with small tolerance)
                    y_diff_current = y_current[curve_a] - y_current[curve_b]
                    y_diff_next = y_next[curve_a] - y_next[curve_b]

                    # A crossing occurs if the sign of the Y difference changes
                    # (i.e., curve A was above B but is now below, or vice versa)
                    TOLERANCE = 0.001
                    if abs(y_diff_current) > TOLERANCE and abs(y_diff_next) > TOLERANCE:
                        a_above_b_current = y_diff_current > 0
                        a_above_b_next = y_diff_next > 0

                        if a_above_b_current != a_above_b_next:
                            # Crossing detected!
                            # Find the divergence point by working backwards
                            divergence_x = find_divergence_point(
                                curves_data, curve_a, curve_b,
                                x_current, sorted_x_positions[:x_idx+1],
                                curve_landing_indices
                            )

                            if divergence_x is not None:
                                # Swap the curve paths from AFTER divergence_x to x_next
                                # We don't swap AT divergence_x because that's where they're together
                                swap_start_idx = sorted_x_positions.index(divergence_x) + 1
                                if swap_start_idx <= x_idx + 1:
                                    swap_start_x = sorted_x_positions[swap_start_idx]
                                    swapped = swap_curve_paths(
                                        curves_data, curve_a, curve_b,
                                        swap_start_x, x_next,
                                        sorted_x_positions, curve_landing_indices
                                    )
                                    if swapped:
                                        swap_count += 1
                                        crossings_fixed_this_pass += 1
                                        # Rebuild indices after swap
                                        curve_landing_indices = build_landing_index(curves_data, curve_names)

        if crossings_fixed_this_pass == 0:
            break  # No more crossings to fix

    # Recalculate peak Y values after all swaps
    recalculate_peak_y_values(curves_data, curve_names)

    return swap_count


def find_divergence_point(curves_data, curve_a, curve_b, current_x, prior_x_positions, curve_landing_indices):
    """
    Find the X position where curve_a and curve_b diverged from a common path.

    We work backwards through prior_x_positions to find where both curves
    were at the same Y position (landing on the same note). This is the point
    from which we should swap their paths.

    Args:
        curves_data: Dict of curve data
        curve_a, curve_b: Names of the two curves that are crossing
        current_x: The X position where we detected the crossing
        prior_x_positions: List of X positions before and including current_x
        curve_landing_indices: Mapping of curve_name -> x_pos -> point_index

    Returns:
        The X position where divergence occurred, or None if not found
    """
    Y_TOLERANCE = 0.01  # How close Y values need to be to be considered "same position"

    # Work backwards through prior X positions
    for x_pos in reversed(prior_x_positions):
        # Check if both curves have landings at this position
        if x_pos not in curve_landing_indices[curve_a] or \
           x_pos not in curve_landing_indices[curve_b]:
            continue

        idx_a = curve_landing_indices[curve_a][x_pos]
        idx_b = curve_landing_indices[curve_b][x_pos]
        y_a = curves_data[curve_a]['points'][idx_a]['y']
        y_b = curves_data[curve_b]['points'][idx_b]['y']

        # If they're at the same Y position, this is where they were together
        if abs(y_a - y_b) < Y_TOLERANCE:
            return x_pos

        # Also check if they're on the same MIDI note
        note_a = curves_data[curve_a]['points'][idx_a].get('note', -1)
        note_b = curves_data[curve_b]['points'][idx_b].get('note', -1)
        if note_a == note_b and note_a != -1:
            return x_pos

    # If we didn't find a clear divergence point, return the earliest position
    # where both curves have landings
    for x_pos in prior_x_positions:
        if x_pos in curve_landing_indices[curve_a] and \
           x_pos in curve_landing_indices[curve_b]:
            return x_pos

    return None


def swap_curve_paths(curves_data, curve_a, curve_b, start_x, end_x, sorted_x_positions, curve_landing_indices):
    """
    Swap the paths of curve_a and curve_b from start_x to end_x (inclusive).

    This swaps all point data (x, y, svgX, svgY, note, noteName) between
    the two curves for all landing points in the specified range.
    Also swaps the associated peak points.

    Args:
        curves_data: Dict of curve data (modified in place)
        curve_a, curve_b: Names of the two curves to swap
        start_x: X position to start swapping (inclusive)
        end_x: X position to end swapping (inclusive)
        sorted_x_positions: All X positions in chronological order
        curve_landing_indices: Mapping of curve_name -> x_pos -> point_index

    Returns:
        True if swap was performed, False otherwise
    """
    # Find the X positions in the swap range
    swap_x_positions = [x for x in sorted_x_positions if start_x <= x <= end_x]

    if not swap_x_positions:
        return False

    points_a = curves_data[curve_a]['points']
    points_b = curves_data[curve_b]['points']

    # For each X position in the range, swap the landing point data
    for x_pos in swap_x_positions:
        if x_pos not in curve_landing_indices[curve_a] or \
           x_pos not in curve_landing_indices[curve_b]:
            continue

        idx_a = curve_landing_indices[curve_a][x_pos]
        idx_b = curve_landing_indices[curve_b][x_pos]

        # Swap the key landing point attributes
        for key in ['y', 'svgY', 'note', 'noteName']:
            points_a[idx_a][key], points_b[idx_b][key] = \
                points_b[idx_b][key], points_a[idx_a][key]

        # Also swap the arc_peak point that precedes this landing (if exists)
        # The arc_peak is typically at idx - 1
        if idx_a > 0 and points_a[idx_a - 1].get('pointType') == 'arc_peak' and \
           idx_b > 0 and points_b[idx_b - 1].get('pointType') == 'arc_peak':
            peak_idx_a = idx_a - 1
            peak_idx_b = idx_b - 1
            for key in ['y', 'svgY', 'noteName']:
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
