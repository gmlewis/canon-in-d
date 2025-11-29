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


def extract_svg_note_centers(svg_file):
    """
    Extract the center coordinates of all note head paths from the SVG file.

    Returns a list of (x, y) tuples sorted by X position (left to right).
    These coordinates are in SVG units and match what Blender imports.

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

    # Sort by X position (left to right, which corresponds to time order)
    centers.sort(key=lambda c: c[0])

    print(f"\nExtracted {len(centers)} note head centers from SVG")
    if centers:
        print(f"  First: ({centers[0][0]:.2f}, {centers[0][1]:.2f})")
        print(f"  Last:  ({centers[-1][0]:.2f}, {centers[-1][1]:.2f})")

    return centers


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

    This matches noteOn events (sorted by time, then by note) with SVG path centers
    (sorted by X position). They should have a 1-to-1 correspondence since each
    note head in the SVG corresponds to exactly one noteOn event.

    Modifies the noteOn events in place, updating their 'svgX' and 'svgY' fields.
    Returns the number of notes matched.
    """
    # Collect all noteOn events with their references
    note_on_events = []
    for track in data:
        if not isinstance(track, list):
            continue
        for event in track:
            if isinstance(event, dict) and event.get('type') == 'noteOn':
                note_on_events.append(event)

    # Sort by time, then by note (for events at the same time)
    note_on_events.sort(key=lambda e: (e.get('time', 0), e.get('note', 0)))

    if len(note_on_events) != len(svg_centers):
        print(f"WARNING: Note count mismatch!")
        print(f"  noteOn events: {len(note_on_events)}")
        print(f"  SVG note heads: {len(svg_centers)}")
        print("  Will match as many as possible...")

    # Match events to SVG centers 1-to-1
    matched = 0
    for i, event in enumerate(note_on_events):
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
    - By default, all curves follow the "bass line" (lowest active note)
    - When extra notes need to be played, curves "split off" to handle them
    - When a curve's note ends (noteOff), it "merges back" to the bass line
    - When merged, multiple curves land on the same note (visually appearing as one blob)

    RULES:
    1. Every curve must have a landing at every noteOn timestamp (no gaps!)
    2. Curves are assigned to notes from lowest to highest pitch
    3. Extra curves (when fewer notes than curves) stack on the lowest note (bass)
    4. When a curve's note ends, it merges back to the bass on the NEXT noteOn
    5. NO Y-ONLY JUMPS: handled by always jumping to a new noteOn (different X)

    Returns a dict: {'curve1': [...], 'curve2': [...], ...}
    Each curve's list contains dicts: {noteName, svgX, svgY, timestamp, note, pointType}
    """
    events = collect_note_events(data)

    if not events:
        return {}, 0, 0

    # Find the end time of the song (last noteOff time)
    end_time = max(e[0] for e in events) + 1.0
    start_time = min(e[0] for e in events)

    # Initialize curve data structure
    curves = {f'curve{i+1}': [] for i in range(max_curves)}
    curve_names = [f'curve{i+1}' for i in range(max_curves)]

    # Build a list of all noteOn timestamps with their active notes at that moment
    # We need to track what notes are active at each noteOn timestamp

    # First, collect all noteOn events with their info
    note_on_events = []  # [(time, note, event), ...]
    for time, event_type, event in events:
        if event_type == 'noteOn':
            note_on_events.append((time, event.get('note'), event))

    # Build a map of note durations: note_num -> [(start_time, end_time), ...]
    # Handle polyphony by tracking multiple instances
    note_instances = {}  # note_num -> [(start, end), ...]
    active_starts = {}   # note_num -> [start_times currently active]

    for time, event_type, event in events:
        note_num = event.get('note')
        if event_type == 'noteOn':
            if note_num not in active_starts:
                active_starts[note_num] = []
            active_starts[note_num].append(time)
        elif event_type == 'noteOff':
            if note_num in active_starts and active_starts[note_num]:
                start = active_starts[note_num].pop(0)  # FIFO
                if note_num not in note_instances:
                    note_instances[note_num] = []
                note_instances[note_num].append((start, time))

    # Function to check if a note is active at a given time
    def is_note_active(note_num, time):
        if note_num not in note_instances:
            return False
        for start, end in note_instances[note_num]:
            if start <= time < end:
                return True
        return False

    # Get all unique noteOn timestamps
    note_on_times = sorted(set(t for t, n, e in note_on_events))

    if not note_on_times:
        return curves, start_time, end_time

    # Find the first noteOn event for the starting position
    first_time = note_on_times[0]
    first_events = [(t, n, e) for t, n, e in note_on_events if t == first_time]
    first_event = min(first_events, key=lambda x: x[1])[2]  # Lowest pitch

    # All curves start at the first note position (timestamp 0.0)
    start_point = {
        'noteName': first_event.get('name', ''),
        'note': first_event.get('note', 0),
        'svgX': first_event.get('svgX', 0),
        'svgY': first_event.get('svgY', 0),
        'timestamp': 0.0,
        'pointType': 'start'
    }
    for curve_name in curves:
        curves[curve_name].append(start_point.copy())

    # Build a lookup from (time, note) -> event
    event_lookup = {}
    for time, note, event in note_on_events:
        event_lookup[(time, note)] = event

    # Track what note each curve was on at the previous timestamp
    # Only add a landing if the curve's note CHANGES or if it's a new noteOn
    prev_curve_notes = {cn: None for cn in curve_names}

    # Process each noteOn timestamp
    for current_time in note_on_times:
        # Find all notes that are active at this timestamp
        # A note is active if it started at or before this time and hasn't ended yet
        active_notes = set()
        for note_num in note_instances:
            if is_note_active(note_num, current_time):
                active_notes.add(note_num)

        # Also add notes that START at this timestamp
        notes_starting_now = set()
        for t, n, e in note_on_events:
            if t == current_time:
                active_notes.add(n)
                notes_starting_now.add(n)

        # Sort active notes by pitch (lowest first)
        sorted_notes = sorted(active_notes)

        # Assign curves to notes:
        # - First N curves go to the N active notes (1 curve per note)
        # - Extra curves (if fewer notes than curves) stack on the LOWEST note (bass)

        num_notes = len(sorted_notes)

        for curve_idx, curve_name in enumerate(curve_names):
            if curve_idx < num_notes:
                # This curve gets its own note
                note_num = sorted_notes[curve_idx]
            else:
                # Extra curve - stack on the bass (lowest note)
                note_num = sorted_notes[0] if sorted_notes else None

            if note_num is None:
                continue

            # Check if we should add a landing point:
            # 1. This note just started (noteOn at this timestamp), OR
            # 2. The curve's assigned note changed from the previous timestamp
            prev_note = prev_curve_notes[curve_name]
            note_just_started = note_num in notes_starting_now
            note_changed = (prev_note != note_num)

            if not note_just_started and not note_changed:
                # Curve stays on the same sustained note - no new landing needed
                continue

            # Find the event for this note at this time (or use active note's original event)
            event = event_lookup.get((current_time, note_num))
            if event is None:
                # Note is active but didn't start at this time - find its original event
                for t, n, e in note_on_events:
                    if n == note_num and t <= current_time:
                        if is_note_active(note_num, current_time):
                            event = e
                            break

            if event is None:
                continue

            # Update tracking
            prev_curve_notes[curve_name] = note_num

            # Add landing point for this curve
            point = {
                'noteName': event.get('name', ''),
                'note': note_num,
                'svgX': event.get('svgX', 0),
                'svgY': event.get('svgY', 0),
                'timestamp': current_time,
                'pointType': 'landing'
            }
            curves[curve_name].append(point)

    # All curves end at end_time, jumping off to the right
    # Use each curve's LAST position for the end point
    for curve_name in curves:
        if curves[curve_name]:
            last_point = curves[curve_name][-1]
            end_point = {
                'noteName': last_point['noteName'],
                'note': last_point['note'],
                'svgX': last_point['svgX'] + END_X_OFFSET,
                'svgY': last_point['svgY'],
                'timestamp': end_time,
                'pointType': 'end'
            }
            curves[curve_name].append(end_point)

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
