#!/usr/bin/env python3
"""
Standalone script to generate note jumping curve data from CanonInD.json.
Outputs a JSON file that can be read by create-curves-from-json.py in Blender.

This script runs OUTSIDE of Blender and does all the data processing.

COORDINATE SYSTEM:
This script reads the SVG file to determine the exact scale factor that
Blender uses when importing SVG files. The formula is:
  blender_scale = (svg_width_mm / svg_viewbox_width) / 1000

Blender imports SVG with Y-axis pointing up (opposite of SVG's convention),
so we negate the Y scale. All coordinates are calculated to match exactly
what Blender produces when importing the SVG file.

CRITICAL GEOMETRY NOTE - X-Y PLANE ONLY:
===========================================
The curve crossover prevention algorithm works STRICTLY in the X-Y plane.
The Z axis (arc height) is IRRELEVANT to this algorithm and can be ignored.

In the X-Y plane:
- X = time (horizontal, left to right)
- Y = pitch (vertical, higher Y = higher musical note in Blender coordinates)

Each curve "jump" between two landing points is a STRAIGHT LINE SEGMENT
in the X-Y plane. The midpoint of this segment is the AVERAGE of the two
endpoints:
  mid_X = (start_X + end_X) / 2
  mid_Y = (start_Y + end_Y) / 2

The Z axis creates the visual "arc" effect (curves go up in Z at midpoint,
then back down to Z=0 at landing), but this is purely cosmetic and does NOT
affect the crossover detection/prevention logic.

INVARIANT TO MAINTAIN:
  curve1.Y <= curve2.Y <= curve3.Y <= ... <= curve7.Y
  at ALL values of X (i.e., at all times)

This ensures curves never visually cross each other in the X-Y plane.
"""

import json
import sys
import os
import re
from xml.etree import ElementTree as ET

# ============================================================================
# Configuration - These values are written to the output JSON for Blender
# ============================================================================
INPUT_JSON_FILE = "CanonInD.json"
OUTPUT_JSON_FILE = "note-jumping-curves.json"
SVG_FILE = "Canon_in_D-single-svg-printing_NoteHeads.svg"

# Blender scene configuration
COLLECTION_NAME = "Note Jumping Curves"
PARENT_NAME = "Note Jumping Curves Parent"
MAX_JUMPING_CURVE_Z_OFFSET = 0.5  # Height of the arc peak between notes

# User's scale factor applied AFTER SVG import in Blender
# Set this to match whatever scale you apply to the imported SVG in Blender
USER_SCALE_FACTOR = 100.0  # You scale the SVG by 100x after import

# How much to offset the final X position (for the "fly off" at end of song)
# This is in SVG units, will be scaled by the computed scale factor
END_X_OFFSET = 500

# These will be computed from SVG file
SVG_SCALE = None       # Computed from SVG viewBox and width
X_SCALE = None         # Same as SVG_SCALE (uniform scaling)
Y_SCALE = None         # Same as SVG_SCALE (positive, for use in formula)
VIEWBOX_HEIGHT = None  # Height of SVG viewBox (for Y flip calculation)


def parse_svg_dimensions(svg_file):
    """
    Parse the SVG file to extract viewBox and width/height.
    Returns (viewbox_width, viewbox_height, width_mm, height_mm).

    Blender's SVG importer calculates scale as:
      scale = (width_in_mm / viewbox_width) / 1000  (to convert mm to meters)
    """
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # Get viewBox
        viewbox = root.get('viewBox', '0 0 1000 1000')
        viewbox_parts = viewbox.split()
        viewbox_width = float(viewbox_parts[2])
        viewbox_height = float(viewbox_parts[3])

        # Get width - parse mm value
        width_str = root.get('width', '1000mm')
        width_match = re.match(r'([\d.]+)\s*mm', width_str)
        if width_match:
            width_mm = float(width_match.group(1))
        else:
            # Try parsing as plain number (assume mm)
            width_mm = float(re.sub(r'[^\d.]', '', width_str))

        # Get height - parse mm value
        height_str = root.get('height', '1000mm')
        height_match = re.match(r'([\d.]+)\s*mm', height_str)
        if height_match:
            height_mm = float(height_match.group(1))
        else:
            height_mm = float(re.sub(r'[^\d.]', '', height_str))

        return viewbox_width, viewbox_height, width_mm, height_mm

    except Exception as e:
        print(f"ERROR: Failed to parse SVG file '{svg_file}': {e}", file=sys.stderr)
        return None, None, None, None


def compute_blender_scale(svg_file):
    """
    Compute the scale factor that Blender uses when importing an SVG.

    Blender's formula: scale = (width_mm / viewbox_width) / 1000
    This converts SVG units to meters.

    Returns (scale, viewbox_width, viewbox_height) or None on error.
    """
    viewbox_width, viewbox_height, width_mm, height_mm = parse_svg_dimensions(svg_file)

    if viewbox_width is None:
        return None

    # Blender's scale calculation
    base_scale = (width_mm / viewbox_width) / 1000.0

    # Apply user's additional scale factor
    scale = base_scale * USER_SCALE_FACTOR

    print(f"\nSVG Coordinate System:")
    print(f"  viewBox: 0 0 {viewbox_width} {viewbox_height}")
    print(f"  width: {width_mm}mm, height: {height_mm}mm")
    print(f"  Blender base scale: {base_scale:.10f} m/unit")
    if USER_SCALE_FACTOR != 1.0:
        print(f"  User scale factor: {USER_SCALE_FACTOR}x")
        print(f"  Final scale: {scale:.10f}")

    return scale, viewbox_width, viewbox_height


def group_by_x_tolerance(items, get_x, tolerance=15.0):
    """
    Group items by X position with tolerance.
    Items within 'tolerance' X units of each other are considered the same chord.

    Returns a list of groups, where each group is sorted by the average X of the group.
    Within each group, items retain their original order.
    """
    if not items:
        return []

    # Sort by X first
    sorted_items = sorted(items, key=get_x)

    groups = []
    current_group = [sorted_items[0]]
    current_group_start_x = get_x(sorted_items[0])

    for item in sorted_items[1:]:
        item_x = get_x(item)
        # Check if this item is within tolerance of the group's starting X
        if item_x - current_group_start_x <= tolerance:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]
            current_group_start_x = item_x

    groups.append(current_group)
    return groups


def extract_svg_note_centers(svg_file):
    """
    Extract the center coordinates of all note head paths from the SVG file.

    Returns a list of (x, y) tuples sorted for chord matching.
    These coordinates are in SVG units and match what Blender imports.

    Sorting strategy:
    - Group paths by X position (within tolerance) to identify chords
    - Sort groups by their minimum X (left to right)
    - Within each chord group, sort by Y (top to bottom = ascending Y)

    This matches notes sorted by (time, descending pitch within chord).

    The note head paths are ellipses drawn with a 'm' (moveto) followed by 'c' (bezier).
    The moveto point is at the left-center of the ellipse. We add half the ellipse
    width (~17.35 units) to get the true center X.
    """
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()
    except Exception as e:
        print(f"ERROR: Failed to parse SVG file '{svg_file}': {e}", file=sys.stderr)
        return None

    centers = []

    for elem in root.iter():
        # Handle namespaced tags
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if tag == 'path':
            d = elem.get('d', '')
            if not d:
                continue

            # Parse the moveto command to get the path start point
            # Format: "m X,Y c ..." or "M X,Y c ..."
            match = re.match(r'm\s+([-\d.]+),([-\d.]+)', d, re.IGNORECASE)
            if not match:
                continue

            start_x = float(match.group(1))
            start_y = float(match.group(2))

            # The note head ellipse is approximately 34.7 units wide
            # The path starts at the left edge, so add half-width to get center X
            # The Y coordinate of the start point is already at the vertical center
            NOTE_HEAD_HALF_WIDTH = 17.35
            center_x = start_x + NOTE_HEAD_HALF_WIDTH
            center_y = start_y

            centers.append((center_x, center_y))

    # Group paths by X position (within tolerance) to identify chords
    # Tolerance of 15 units handles bass/treble alignment differences
    chord_groups = group_by_x_tolerance(centers, lambda c: c[0], tolerance=15.0)

    # Build final sorted list:
    # - Groups sorted by minimum X (left to right)
    # - Within each group, sorted by Y (ascending = top to bottom on page)
    sorted_centers = []
    for group in chord_groups:
        # Sort within chord by Y (ascending Y = top to bottom = high pitch to low pitch)
        group_sorted = sorted(group, key=lambda c: c[1])
        sorted_centers.extend(group_sorted)

    print(f"\nExtracted {len(sorted_centers)} note head centers from SVG")
    if sorted_centers:
        print(f"  First: ({sorted_centers[0][0]:.2f}, {sorted_centers[0][1]:.2f})")
        print(f"  Last:  ({sorted_centers[-1][0]:.2f}, {sorted_centers[-1][1]:.2f})")

    return sorted_centers


def svg_to_blender_x(svg_x):
    """Convert SVG X coordinate to Blender X coordinate."""
    return svg_x * X_SCALE


def svg_to_blender_y(svg_y):
    """
    Convert SVG Y coordinate to Blender Y coordinate.

    Blender flips the Y axis when importing SVG:
    - SVG has Y=0 at top, increasing downward
    - Blender has Y=0 at bottom, increasing upward

    Formula: blender_y = (viewbox_height - svg_y) * scale
    """
    return (VIEWBOX_HEIGHT - svg_y) * Y_SCALE


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


def compute_max_concurrent_notes(data):
    """
    Compute the maximum number of concurrently-playing notes.
    Returns max_concurrent count.
    """
    events = []

    for track in data:
        if not isinstance(track, list):
            continue

        for event in track:
            if not isinstance(event, dict):
                continue

            event_type = event.get('type')
            time = event.get('time')

            if time is None:
                continue

            if event_type == 'noteOn':
                events.append((time, 1, event))
            elif event_type == 'noteOff':
                events.append((time, -1, event))

    events.sort(key=lambda x: (x[0], x[1]))

    current_count = 0
    max_count = 0
    max_time = 0

    for time, delta, event in events:
        current_count += delta
        if current_count > max_count:
            max_count = current_count
            max_time = time

    return max_count, max_time


def collect_note_events(data):
    """
    Collect all noteOn and noteOff events from the JSON data.
    Returns a list of (time, event_type, event) tuples sorted by time.
    """
    events = []

    for track in data:
        if not isinstance(track, list):
            continue

        for event in track:
            if not isinstance(event, dict):
                continue

            event_type = event.get('type')
            time = event.get('time')

            if time is None:
                continue

            if event_type in ('noteOn', 'noteOff'):
                events.append((time, event_type, event))

    # Sort by time, with noteOff before noteOn at same time
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'noteOff' else 1))

    return events


def assign_svg_coords_to_notes(data, svg_centers):
    """
    Assign SVG coordinates from the extracted SVG path centers to noteOn events.

    This matches noteOn events with SVG path centers using chord-aware sorting:
    - Notes grouped by timestamp, SVG paths grouped by X (within tolerance)
    - Within each chord, matched by pitch order (high to low) = Y order (top to bottom)

    Modifies the noteOn events in place, updating their 'svgX' and 'svgY' fields.
    Returns the number of notes matched.
    """
    from collections import defaultdict

    # Collect all noteOn events with their references
    note_on_events = []
    for track in data:
        if not isinstance(track, list):
            continue
        for event in track:
            if isinstance(event, dict) and event.get('type') == 'noteOn':
                note_on_events.append(event)

    # Group notes by timestamp to identify chords
    time_groups = defaultdict(list)
    for note in note_on_events:
        time_groups[note.get('time', 0)].append(note)

    # Build final sorted list:
    # - Groups sorted by time (chronological)
    # - Within each group, sorted by DESCENDING note (higher pitch first)
    #   This matches SVG Y order where lower Y = higher pitch
    sorted_notes = []
    for time in sorted(time_groups.keys()):
        group = time_groups[time]
        group_sorted = sorted(group, key=lambda n: -n.get('note', 0))  # Descending pitch
        sorted_notes.extend(group_sorted)

    if len(sorted_notes) != len(svg_centers):
        print(f"WARNING: Note count mismatch!")
        print(f"  noteOn events: {len(sorted_notes)}")
        print(f"  SVG note heads: {len(svg_centers)}")
        print("  Will match as many as possible...")

    # Match events to SVG centers 1-to-1
    matched = 0
    for i, event in enumerate(sorted_notes):
        if i < len(svg_centers):
            svg_x, svg_y = svg_centers[i]
            event['svgX'] = svg_x
            event['svgY'] = svg_y
            matched += 1

    print(f"  Matched {matched} noteOn events to SVG coordinates")
    return matched


def build_curve_data(data, max_curves):
    """
    Build the curve data dictionary based on note events.

    CONCEPT:
    - All 7 curves bounce continuously from start to end (no waiting!)
    - Curves are assigned to notes maintaining the INVARIANT:
      curve1.Y <= curve2.Y <= ... <= curveN.Y at ALL times (Blender Y)
    - Since higher svgY = lower Blender Y, this means:
      curve1.svgY >= curve2.svgY >= ... >= curveN.svgY
    - This prevents curves from ever crossing each other visually

    ALGORITHM - ONE CURVE AT A TIME, BACKWARDS:

    We trace each curve COMPLETELY before moving to the next curve.
    This simplifies the constraint: when tracing curveN, we only need to
    ensure it stays >= curves 1 through N-1 (which are already fixed).

    For each curve (1 through 7):
    1. Start from the END (fly-out point)
    2. Work backwards through all landing timestamps
    3. At each timestamp, choose the landing position with highest svgY
       that is ALSO >= the svgY of all lower-numbered curves at that time
       (considering their entire arc trajectory, not just landings)
    4. Continue to the START (fly-in point)

    Returns a dict: {'curve1': [...], 'curve2': [...], ...}
    Each curve's list contains dicts: {noteName, svgX, svgY, timestamp, note, pointType}
    """
    events = collect_note_events(data)

    if not events:
        return {}, 0, 0

    end_time = max(e[0] for e in events) + 1.0
    start_time = min(e[0] for e in events)

    curve_names = [f'curve{i+1}' for i in range(max_curves)]

    # Collect all noteOn events
    note_on_events = []
    for time, event_type, event in events:
        if event_type == 'noteOn':
            note_on_events.append((time, event.get('note'), event))

    # Build note duration map: note_num -> [(start, end), ...]
    note_instances = {}
    active_starts = {}
    for time, event_type, event in events:
        note_num = event.get('note')
        if event_type == 'noteOn':
            if note_num not in active_starts:
                active_starts[note_num] = []
            active_starts[note_num].append(time)
        elif event_type == 'noteOff':
            if note_num in active_starts and active_starts[note_num]:
                start_t = active_starts[note_num].pop(0)
                if note_num not in note_instances:
                    note_instances[note_num] = []
                note_instances[note_num].append((start_t, time))

    # Build lookup from (time, note) -> event
    event_lookup = {}
    for time, note, event in note_on_events:
        event_lookup[(time, note)] = event

    note_on_times = sorted(set(t for t, n, e in note_on_events))
    if not note_on_times:
        return {cn: [] for cn in curve_names}, start_time, end_time

    first_time = note_on_times[0]
    first_events_list = [(t, n, e) for t, n, e in note_on_events if t == first_time]
    first_event = min(first_events_list, key=lambda x: x[1])[2]

    # === HELPER: Get svgY of a curve at any time via linear interpolation ===
    def get_curve_svgY_at_time(curve_landings, t):
        """
        Get the svgY of a curve at time t, using linear interpolation between landings.
        curve_landings is a list of (time, note, svgX, svgY, noteName) sorted by time.
        """
        if not curve_landings:
            return None

        # Find bracketing landings
        prev_landing = None
        next_landing = None
        for landing in curve_landings:
            if landing[0] <= t:
                prev_landing = landing
            if landing[0] >= t and next_landing is None:
                next_landing = landing

        if prev_landing is None:
            return curve_landings[0][3]  # Before first landing
        if next_landing is None:
            return curve_landings[-1][3]  # After last landing
        if prev_landing[0] == next_landing[0]:
            return prev_landing[3]

        # Linear interpolation
        ratio = (t - prev_landing[0]) / (next_landing[0] - prev_landing[0])
        return prev_landing[3] + ratio * (next_landing[3] - prev_landing[3])

    # === HELPER: Check if a position is valid for curveN given curves 1..N-1 ===
    def is_position_valid(curve_num, origin_svgY, origin_time, dest_svgY, dest_time,
                          completed_curves):
        """
        Check if assigning curveN to go from origin to dest maintains the invariant.

        The invariant: curveN.svgY <= curve(N-1).svgY <= ... <= curve1.svgY
        (Remember: higher svgY = lower Blender Y)

        We need to check this at ALL times between origin_time and dest_time,
        not just at the landing points.
        """
        if not completed_curves:
            return True  # curve1 has no constraints

        # Check at multiple time points between origin and dest
        # The arc peak is at the midpoint, so we especially need to check there
        check_times = [origin_time]
        if origin_time != dest_time:
            mid_time = (origin_time + dest_time) / 2
            check_times.extend([
                origin_time + (dest_time - origin_time) * 0.25,
                mid_time,
                origin_time + (dest_time - origin_time) * 0.75,
                dest_time
            ])

        for t in check_times:
            # Interpolate this curve's svgY at time t
            if origin_time == dest_time:
                my_svgY = origin_svgY
            else:
                ratio = (t - origin_time) / (dest_time - origin_time)
                my_svgY = origin_svgY + ratio * (dest_svgY - origin_svgY)

            # Check against all completed curves
            for prev_curve_name, prev_landings in completed_curves.items():
                prev_svgY = get_curve_svgY_at_time(prev_landings, t)
                if prev_svgY is None:
                    continue

                # curveN must have svgY <= prev_curve's svgY
                # (lower svgY = higher Blender Y = curveN should be above prev curves)
                if my_svgY > prev_svgY + 0.001:  # Small tolerance
                    return False

        return True    # === STEP 1: Build the list of all landing timestamps and available notes ===

    # For each noteOn time, what notes are available?
    notes_at_time = {}  # time -> [(note, svgX, svgY, noteName, end_time), ...]
    for t, note, event in note_on_events:
        if t not in notes_at_time:
            notes_at_time[t] = []
        # Find end time for this note
        end_t = end_time
        for s, et in note_instances.get(note, []):
            if s == t:
                end_t = et
                break
        notes_at_time[t].append((
            note,
            event.get('svgX', 0),
            event.get('svgY', 0),
            event.get('name', ''),
            end_t
        ))

    # Sort notes at each time by svgY descending (highest svgY = curve1's preferred slot)
    for t in notes_at_time:
        notes_at_time[t].sort(key=lambda x: -x[2])

    # === STEP 2: Trace each curve one at a time ===

    completed_curves = {}  # curve_name -> [(time, note, svgX, svgY, noteName), ...]

    for curve_idx, curve_name in enumerate(curve_names):
        # Trace this curve from END to START
        curve_landings = []

        # Start with the last note (we'll work backwards)
        # Find the last noteOn time
        last_time = note_on_times[-1]

        # Current state: which note is this curve on, and when does it end?
        # We work backwards, so "current" means "future" in forward time
        current_note = None
        current_svgY = None
        current_end_time = end_time

        # Process timestamps from end to start
        times_to_process = sorted(note_on_times, reverse=True)

        for current_time in times_to_process:
            notes_available = notes_at_time.get(current_time, [])
            if not notes_available:
                continue

            # Do we need to land at this time?
            # Yes, if:
            # 1. This is our first assignment (current_note is None), OR
            # 2. Our current note started at this time (we need an origin)

            need_landing = False
            if current_note is None:
                need_landing = True
            else:
                # Check if current_note started at current_time
                for note, svgX, svgY, noteName, end_t in notes_available:
                    if note == current_note:
                        need_landing = True
                        break

            if not need_landing:
                # Check if our current note was still available (active) at this time
                # If not, we need to find where we were before
                note_was_active = False
                for s, e in note_instances.get(current_note, []):
                    if s <= current_time < e:
                        note_was_active = True
                        break
                if not note_was_active:
                    need_landing = True

            if not need_landing:
                continue

            # Find the best landing position at this time
            # We want the highest svgY that maintains the invariant
            best_landing = None

            for note, svgX, svgY, noteName, end_t in notes_available:
                # Check if this position is valid
                if current_note is None:
                    # First landing (at end of song), just pick highest svgY
                    # that's valid against completed curves
                    dest_svgY = svgY
                    dest_time = current_time
                    # For the "destination" check, use the same position
                    if is_position_valid(curve_idx, svgY, current_time, svgY, current_time,
                                        completed_curves):
                        if best_landing is None or svgY > best_landing[2]:
                            best_landing = (note, svgX, svgY, noteName, end_t)
                else:
                    # Check if arc from this origin to current destination is valid
                    # Origin: (svgY at current_time)
                    # Destination: (current_svgY at the time we land there)
                    # Note: curve_landings is in reverse time order (newest first),
                    # so we need to find the SMALLEST time > current_time (the immediate next landing)
                    dest_time = None
                    for landing in curve_landings:
                        if landing[0] > current_time:
                            # This is a candidate - but we want the smallest one
                            if dest_time is None or landing[0] < dest_time:
                                dest_time = landing[0]
                    if dest_time is None:
                        dest_time = current_time

                    if is_position_valid(curve_idx, svgY, current_time, current_svgY, dest_time,
                                        completed_curves):
                        if best_landing is None or svgY > best_landing[2]:
                            best_landing = (note, svgX, svgY, noteName, end_t)

            if best_landing is None:
                # No valid position found - fall back to highest available
                # This shouldn't happen if the algorithm is correct
                print(f"WARNING: No valid position for {curve_name} at t={current_time}")
                best_landing = notes_available[0]

            note, svgX, svgY, noteName, end_t = best_landing
            curve_landings.append((current_time, note, svgX, svgY, noteName))
            current_note = note
            current_svgY = svgY
            current_end_time = end_t

        # Reverse to get chronological order
        curve_landings.reverse()
        completed_curves[curve_name] = curve_landings

    # === STEP 3: Build final curve data ===

    curves = {}
    for cn in curve_names:
        points = []
        landings = completed_curves.get(cn, [])

        # Add start point (fly-in)
        start_point = {
            'noteName': first_event.get('name', ''),
            'note': first_event.get('note', 0),
            'svgX': first_event.get('svgX', 0),
            'svgY': first_event.get('svgY', 0),
            'timestamp': 0.0,
            'pointType': 'start'
        }
        points.append(start_point)

        # Add landing points
        for t, note, svgX, svgY, noteName in landings:
            if t > 0:
                point = {
                    'noteName': noteName,
                    'note': note,
                    'svgX': svgX,
                    'svgY': svgY,
                    'timestamp': t,
                    'pointType': 'landing'
                }
                points.append(point)

        # Add end point (fly-off)
        if points:
            last_point = points[-1]
            end_point = {
                'noteName': last_point['noteName'],
                'note': last_point['note'],
                'svgX': last_point['svgX'] + END_X_OFFSET,
                'svgY': last_point['svgY'],
                'timestamp': end_time,
                'pointType': 'end'
            }
            points.append(end_point)

        curves[cn] = points

    return curves, start_time, end_time


def generate_bezier_points(curves, z_offset):
    """
    Generate the actual bezier control points for each curve.

    For each curve, creates:
    - A "fly-in" start point hovering at Z=z_offset, off-screen to the left
    - Peak points (Z=z_offset) at midpoints between landings
    - Landing points (Z=0) at each note position
    - A "fly-off" end point hovering at Z=z_offset, off-screen to the right

    Returns a dict with bezier point data ready for Blender.
    """
    bezier_curves = {}

    # Offset for fly-in/fly-off points (in Blender units, already scaled)
    FLY_OFFSET = 1.0  # 1 meter off-screen

    for curve_name, landing_points in curves.items():
        if len(landing_points) < 2:
            continue

        bezier_points = []

        # Get the first landing point info for the fly-in
        first_landing = landing_points[0]
        first_x = svg_to_blender_x(first_landing['svgX'])
        first_y = svg_to_blender_y(first_landing['svgY'])

        # Add fly-in start point (hovering at z_offset, WAY off to the left at x=-1)
        fly_in_start = {
            'x': -1.0,  # Absolute position: way off-screen left
            'y': first_y,
            'z': z_offset,  # Hovering, not on the ground
            'type': 'fly_in',
            'noteName': f"fly-in -> {first_landing['noteName']}",
            'note': None,
            'timestamp': first_landing['timestamp'] - 1.0,
            'pointType': 'fly_in_start',
            'svgX': -100,  # Off-screen in SVG coords too
            'svgY': first_landing['svgY']
        }
        bezier_points.append(fly_in_start)

        # Add arc peak between fly-in and first landing
        fly_in_peak = {
            'x': first_x - (FLY_OFFSET / 2.0),
            'y': first_y,
            'z': z_offset,  # Peak at same height as fly-in
            'type': 'peak',
            'noteName': f"fly-in peak -> {first_landing['noteName']}",
            'note': None,
            'timestamp': first_landing['timestamp'] - 0.5,
            'pointType': 'fly_in_peak',
            'svgX': first_landing['svgX'] - (FLY_OFFSET / 2.0 / X_SCALE),
            'svgY': first_landing['svgY']
        }
        bezier_points.append(fly_in_peak)

        # Track the last landing we actually added (for peak calculations)
        last_added_landing = None

        for i, point in enumerate(landing_points):
            current_x = svg_to_blender_x(point['svgX'])
            current_y = svg_to_blender_y(point['svgY'])

            # Skip this landing if it's at the exact same position as the last added landing
            # (avoids duplicate consecutive points at same x,y,z)
            if last_added_landing is not None:
                same_as_prev = (abs(last_added_landing['svgX'] - point['svgX']) < 0.01 and
                                abs(last_added_landing['svgY'] - point['svgY']) < 0.01)
                if same_as_prev:
                    continue

            # Add peak point between last added landing and this one (if positions differ)
            if last_added_landing is not None:
                mid_svg_x = (last_added_landing['svgX'] + point['svgX']) / 2.0
                mid_svg_y = (last_added_landing['svgY'] + point['svgY']) / 2.0
                peak = {
                    'x': svg_to_blender_x(mid_svg_x),
                    'y': svg_to_blender_y(mid_svg_y),
                    'z': z_offset,
                    'type': 'peak',
                    'noteName': f"{last_added_landing['noteName']} -> {point['noteName']}",
                    'note': None,
                    'timestamp': (last_added_landing['timestamp'] + point['timestamp']) / 2.0,
                    'pointType': 'arc_peak',
                    'svgX': mid_svg_x,
                    'svgY': mid_svg_y
                }
                bezier_points.append(peak)

            # Landing point (on the note, Z=0)
            landing = {
                'x': current_x,
                'y': current_y,
                'z': 0.0,
                'type': 'landing',
                'noteName': point['noteName'],
                'note': point['note'],
                'timestamp': point['timestamp'],
                'pointType': point['pointType'],
                'svgX': point['svgX'],
                'svgY': point['svgY']
            }
            bezier_points.append(landing)
            last_added_landing = point  # Track for next iteration

        # Get the last landing point info for the fly-off
        last_landing = landing_points[-1]
        last_x = svg_to_blender_x(last_landing['svgX'])
        last_y = svg_to_blender_y(last_landing['svgY'])

        # Add arc peak between last landing and fly-off
        fly_off_peak = {
            'x': last_x + (FLY_OFFSET / 2.0),
            'y': last_y,
            'z': z_offset,
            'type': 'peak',
            'noteName': f"{last_landing['noteName']} -> fly-off peak",
            'note': None,
            'timestamp': last_landing['timestamp'] + 0.5,
            'pointType': 'fly_off_peak',
            'svgX': last_landing['svgX'] + (FLY_OFFSET / 2.0 / X_SCALE),
            'svgY': last_landing['svgY']
        }
        bezier_points.append(fly_off_peak)

        # Add fly-off end point (hovering at z_offset, WAY off to the right)
        fly_off_end = {
            'x': last_x + 5.0,  # 5 meters past last landing - way off-screen
            'y': last_y,
            'z': z_offset,  # Hovering, not on the ground
            'type': 'fly_off',
            'noteName': f"{last_landing['noteName']} -> fly-off",
            'note': None,
            'timestamp': last_landing['timestamp'] + 1.0,
            'pointType': 'fly_off_end',
            'svgX': last_landing['svgX'] + 500,  # Way off-screen in SVG coords
            'svgY': last_landing['svgY']
        }
        bezier_points.append(fly_off_end)

        bezier_curves[curve_name] = {
            'landingCount': len(landing_points),
            'bezierPointCount': len(bezier_points),
            'points': bezier_points
        }

    return bezier_curves


def main():
    global X_SCALE, Y_SCALE, SVG_SCALE, VIEWBOX_HEIGHT

    print("=" * 70)
    print("Note Jumping Curves Generator (Standalone)")
    print("=" * 70)

    # Step 0: Compute scale from SVG file and extract note head positions
    print(f"\nReading SVG file '{SVG_FILE}'...")
    result = compute_blender_scale(SVG_FILE)
    if result is None:
        print("ERROR: Could not parse SVG file. Aborting.")
        return 1

    scale, viewbox_width, viewbox_height = result
    X_SCALE = scale
    Y_SCALE = scale  # Same scale, but Y formula uses (viewbox_height - svg_y)
    VIEWBOX_HEIGHT = viewbox_height
    SVG_SCALE = scale  # Store for reference

    print(f"  Scale = {scale:.10f}")
    print(f"  viewBox height = {viewbox_height} (for Y flip)")
    print(f"  Formula: blender_x = svg_x * scale")
    print(f"  Formula: blender_y = (viewbox_height - svg_y) * scale")

    # Extract note head centers from SVG
    svg_centers = extract_svg_note_centers(SVG_FILE)
    if svg_centers is None or len(svg_centers) == 0:
        print("ERROR: Could not extract note positions from SVG. Aborting.")
        return 1

    # Step 1: Load JSON data
    print(f"\nLoading '{INPUT_JSON_FILE}'...")
    data = load_json_data(INPUT_JSON_FILE)

    if data is None:
        print("Aborting due to JSON load failure.")
        return 1

    print(f"Successfully loaded JSON with {len(data)} tracks.")

    # Step 1b: Assign SVG coordinates to noteOn events
    print("\nMatching notes to SVG positions...")
    assign_svg_coords_to_notes(data, svg_centers)

    # Step 2: Compute MAX_CURVES
    print("\nAnalyzing note overlaps...")
    max_curves, max_time = compute_max_concurrent_notes(data)
    print(f"MAX_CURVES = {max_curves} (maximum concurrent notes)")
    print(f"  Peak concurrency occurs at time {max_time:.2f}s")

    # Step 3: Build curve data
    print("\nBuilding curve data...")
    curves, start_time, end_time = build_curve_data(data, max_curves)

    # Print summary
    print(f"\nCurve data summary:")
    print(f"  Song duration: {start_time:.2f}s to {end_time - 1:.2f}s")
    total_points = 0
    for curve_name in sorted(curves.keys(), key=lambda x: int(x.replace('curve', ''))):
        points = curves[curve_name]
        total_points += len(points)
        print(f"  {curve_name}: {len(points)} landing points")
        if points:
            print(f"    First: t={points[0]['timestamp']:.2f}s, note={points[0]['noteName']} ({points[0]['note']})")
            print(f"    Last:  t={points[-1]['timestamp']:.2f}s, note={points[-1]['noteName']} ({points[-1]['note']})")

    print(f"\n  Total landing points across all curves: {total_points}")

    # Step 4: Generate bezier control points
    print("\nGenerating bezier control points...")
    bezier_curves = generate_bezier_points(curves, MAX_JUMPING_CURVE_Z_OFFSET)

    total_bezier = sum(c['bezierPointCount'] for c in bezier_curves.values())
    print(f"  Total bezier points: {total_bezier}")

    # Step 5: Build output JSON
    output = {
        '_comment': 'Generated by gen-note-jumping-curves-to-json.py - edit values below to customize',
        'config': {
            'collectionName': COLLECTION_NAME,
            'parentName': PARENT_NAME,
            'maxJumpingCurveZOffset': MAX_JUMPING_CURVE_Z_OFFSET,
            'scale': SVG_SCALE,
            'viewboxHeight': VIEWBOX_HEIGHT,
            'userScaleFactor': USER_SCALE_FACTOR,
            'svgFile': SVG_FILE,
            'curveResolution': 12,
            'handleType': 'AUTO',
            '_formulaX': 'blender_x = svg_x * scale',
            '_formulaY': 'blender_y = (viewboxHeight - svg_y) * scale'
        },
        'metadata': {
            'sourceFile': INPUT_JSON_FILE,
            'maxConcurrentNotes': max_curves,
            'peakConcurrencyTime': max_time,
            'songStartTime': start_time,
            'songEndTime': end_time - 1,
            'totalLandingPoints': total_points,
            'totalBezierPoints': total_bezier
        },
        'curves': bezier_curves
    }

    # Step 6: Write output JSON
    print(f"\nWriting output to '{OUTPUT_JSON_FILE}'...")
    try:
        with open(OUTPUT_JSON_FILE, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Successfully wrote {os.path.getsize(OUTPUT_JSON_FILE):,} bytes")
    except Exception as e:
        print(f"ERROR: Failed to write output file: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("Generation complete!")
    print(f"  Output file: {OUTPUT_JSON_FILE}")
    print(f"  Curves: {len(bezier_curves)}")
    print(f"  Total bezier points: {total_bezier}")
    print("\nNext step: Run 'create-curves-from-json.py' in Blender")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
