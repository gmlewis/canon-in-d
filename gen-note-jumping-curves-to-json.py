#!/usr/bin/env python3
"""
Standalone script to generate note jumping curve data from CanonInD.json.
Outputs a JSON file that can be read by create-curves-from-json.py in Blender.

This script runs OUTSIDE of Blender and does all the data processing.

==============================================================================
CRITICAL INSTRUCTION FOR AI AGENTS (Copilot, Claude, GPT, etc.):
==============================================================================

>>> NEVER WORK WITH RAW SVG COORDINATES DIRECTLY! <<<

The SVG coordinate system has Y increasing DOWNWARD (Y=0 at top), which is
the OPPOSITE of musical intuition and Blender's coordinate system.

This script provides coordinate transformation functions that MUST be used:

    svg_to_blender(svgX, svgY) -> (blenderX, blenderY)
    blender_to_svg(blenderX, blenderY) -> (svgX, svgY)

ALL internal algorithm logic uses BLENDER COORDINATES where:
    - X = time (increases left to right)
    - Y = pitch (increases upward - HIGHER Y = HIGHER MUSICAL NOTE)

The invariant to maintain is simple and intuitive in Blender coordinates:
    curve1.Y <= curve2.Y <= curve3.Y <= ... <= curve7.Y

This means curve1 is always at the BOTTOM (lowest pitch notes) and
curve7 is always at the TOP (highest pitch notes).

SVG coordinates are ONLY used:
    1. When reading from the SVG file
    2. When writing to the output JSON (for Blender to consume)

NEVER compare svgY values directly. NEVER think about "higher svgY means
lower pitch" - just use Blender coordinates and think naturally.

==============================================================================
COORDINATE SYSTEM DETAILS:
==============================================================================

This script reads the SVG file to determine the exact scale factor that
Blender uses when importing SVG files. The formula is:
  blender_scale = (svg_width_mm / svg_viewbox_width) / 1000

Blender imports SVG with Y-axis pointing up (opposite of SVG's convention).
The transformation is:
  blender_x = svg_x * scale
  blender_y = (viewbox_height - svg_y) * scale

X-Y PLANE GEOMETRY:
Each curve "jump" between two landing points is a STRAIGHT LINE SEGMENT
in the X-Y plane. The midpoint of this segment is the AVERAGE of the two
endpoints:
  mid_X = (start_X + end_X) / 2
  mid_Y = (start_Y + end_Y) / 2

The Z axis creates the visual "arc" effect (curves go up in Z at midpoint,
then back down to Z=0 at landing), but Z is IRRELEVANT to crossover detection.

INVARIANT (in Blender coordinates):
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


# ============================================================================
# COORDINATE TRANSFORMATION FUNCTIONS
# ============================================================================
# USE THESE FUNCTIONS! Do not manually work with SVG coordinates!
# ============================================================================

def svg_to_blender(svgX, svgY):
    """
    Transform SVG coordinates to Blender coordinates.

    USE THIS FUNCTION for all coordinate transformations!

    In Blender coordinates:
    - X increases left to right (same as SVG)
    - Y increases UPWARD (opposite of SVG) - higher Y = higher pitch

    Returns: (blenderX, blenderY) tuple
    """
    blenderX = svgX * X_SCALE
    blenderY = (VIEWBOX_HEIGHT - svgY) * Y_SCALE
    return (blenderX, blenderY)


def blender_to_svg(blenderX, blenderY):
    """
    Transform Blender coordinates back to SVG coordinates.

    Use this when writing output that needs SVG coordinates.

    Returns: (svgX, svgY) tuple
    """
    svgX = blenderX / X_SCALE
    svgY = VIEWBOX_HEIGHT - (blenderY / Y_SCALE)
    return (svgX, svgY)


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

    This matches noteOn events with SVG path centers using CHORD-BY-CHORD matching:
    1. Group MIDI notes by timestamp (chord groups)
    2. Group SVG centers by X position (chord groups)
    3. Match chord groups 1-to-1 in order (first MIDI chord → first SVG chord, etc.)
    4. Within each matched chord pair, match by pitch/Y order:
       - MIDI notes sorted by descending pitch (highest first)
       - SVG centers sorted by ascending Y (top first = highest pitch)

    This handles cases where SVG has extra notes (e.g., ornaments not in MIDI)
    by ensuring each chord's notes are matched correctly regardless of global count.

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

    # Group notes by timestamp to identify MIDI chords
    time_groups = defaultdict(list)
    for note in note_on_events:
        time_groups[note.get('time', 0)].append(note)

    # Build list of MIDI chord groups (sorted by time)
    # Each chord group contains notes sorted by descending pitch (highest first)
    midi_chord_groups = []
    for time in sorted(time_groups.keys()):
        group = time_groups[time]
        group_sorted = sorted(group, key=lambda n: -n.get('note', 0))  # Descending pitch
        midi_chord_groups.append((time, group_sorted))

    # Group SVG centers by X position to identify SVG chords
    svg_chord_groups = group_by_x_tolerance(svg_centers, lambda c: c[0], tolerance=15.0)
    # Within each SVG chord, sort by ascending Y (top to bottom = high to low pitch)
    for i, group in enumerate(svg_chord_groups):
        svg_chord_groups[i] = sorted(group, key=lambda c: c[1])

    print(f"  MIDI chord groups: {len(midi_chord_groups)}")
    print(f"  SVG chord groups: {len(svg_chord_groups)}")

    if len(midi_chord_groups) != len(svg_chord_groups):
        print(f"  WARNING: Chord group count mismatch!")
        print(f"    Will match as many chord groups as possible...")

    # Match chord groups 1-to-1
    matched = 0
    total_chord_mismatches = 0
    for i, (midi_time, midi_notes) in enumerate(midi_chord_groups):
        if i >= len(svg_chord_groups):
            print(f"  WARNING: Ran out of SVG chord groups at MIDI chord {i} (time={midi_time:.3f})")
            break

        svg_notes = svg_chord_groups[i]

        # Within this chord, match notes 1-to-1 by pitch/Y order
        if len(midi_notes) != len(svg_notes):
            total_chord_mismatches += 1
            # Only print details for first few mismatches
            if total_chord_mismatches <= 3:
                print(f"  WARNING: Chord size mismatch at time={midi_time:.3f}:")
                print(f"    MIDI notes: {len(midi_notes)}, SVG notes: {len(svg_notes)}")

        for j, note_event in enumerate(midi_notes):
            if j < len(svg_notes):
                svg_x, svg_y = svg_notes[j]
                note_event['svgX'] = svg_x
                note_event['svgY'] = svg_y
                matched += 1

    if total_chord_mismatches > 3:
        print(f"  ... and {total_chord_mismatches - 3} more chord size mismatches")

    print(f"  Matched {matched} noteOn events to SVG coordinates")
    return matched


def build_curve_data(data, max_curves):
    """
    Build the curve data dictionary based on note events.

    ===========================================================================
    ALL LOGIC USES BLENDER COORDINATES (bY) WHERE:
        - Higher bY = Higher pitch (more intuitive!)
        - curve1.bY <= curve2.bY <= ... <= curve7.bY (curve1 at bottom, curve7 at top)

    We convert SVG coordinates to Blender coordinates IMMEDIATELY and work
    entirely in Blender coordinates. SVG coordinates are only used at output.
    ===========================================================================

    ALGORITHM - ONE CURVE AT A TIME, BACKWARDS:

    We trace each curve COMPLETELY before moving to the next curve.
    curve1 is traced first and gets the LOWEST available bY positions.
    curve2 is traced second and must stay >= curve1's bY at all times.
    And so on up to curve7 which gets the HIGHEST bY positions.

    For each curve (1 through 7):
    1. Start from the END (fly-out point)
    2. Work backwards through all landing timestamps
    3. At each timestamp, choose the landing position with LOWEST bY
       that is ALSO >= the bY of all lower-numbered curves at that time
    4. Continue to the START (fly-in point)

    Returns a dict: {'curve1': [...], 'curve2': [...], ...}
    Each curve's list contains dicts with both SVG and Blender coordinates.
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

    # === HELPER: Get Blender Y of a curve at any time via linear interpolation ===
    def get_curve_bY_at_time(curve_landings, t):
        """
        Get the Blender Y (bY) of a curve at time t, using linear interpolation.
        curve_landings is a list of (time, note, svgX, svgY, bY, noteName) sorted by time.

        Uses BLENDER coordinates where higher bY = higher pitch.
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
            return curve_landings[0][4]  # bY is at index 4
        if next_landing is None:
            return curve_landings[-1][4]
        if prev_landing[0] == next_landing[0]:
            return prev_landing[4]

        # Linear interpolation
        ratio = (t - prev_landing[0]) / (next_landing[0] - prev_landing[0])
        return prev_landing[4] + ratio * (next_landing[4] - prev_landing[4])

    # === HELPER: Check if a position is valid for curveN given curves 1..N-1 ===
    def is_position_valid(curve_num, origin_bY, origin_time, dest_bY, dest_time,
                          completed_curves):
        """
        Check if assigning curveN to go from origin to dest maintains the invariant.

        INVARIANT (in Blender coordinates):
            curveN.bY >= curve(N-1).bY >= ... >= curve1.bY

        In other words, higher-numbered curves must have HIGHER (or equal) bY
        at all times. This is intuitive: curve7 is at the top, curve1 at bottom.

        We check this at multiple times between origin and dest to catch
        crossings that might occur during the arc.
        """
        if not completed_curves:
            return True  # curve1 has no constraints

        # Check at multiple time points between origin and dest
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
            # Interpolate this curve's bY at time t
            if origin_time == dest_time:
                my_bY = origin_bY
            else:
                ratio = (t - origin_time) / (dest_time - origin_time)
                my_bY = origin_bY + ratio * (dest_bY - origin_bY)

            # Check against all completed (lower-numbered) curves
            for prev_curve_name, prev_landings in completed_curves.items():
                prev_bY = get_curve_bY_at_time(prev_landings, t)
                if prev_bY is None:
                    continue

                # curveN must have bY >= prev_curve's bY (higher or equal)
                # Using small tolerance for floating point
                if my_bY < prev_bY - 0.001:
                    return False

        return True

    # === STEP 1: Build the list of all landing timestamps and available notes ===
    # Convert to Blender coordinates IMMEDIATELY - never use raw SVG coords in logic!

    # For each noteOn time, what notes are available?
    # Format: (note, svgX, svgY, bY, noteName, end_time)
    # We keep both svgY (for output) and bY (for all logic)
    notes_at_time = {}
    skipped_no_svg = 0
    for t, note, event in note_on_events:
        svgX = event.get('svgX', 0)
        svgY = event.get('svgY', 0)

        # Skip notes that don't have valid SVG coordinates
        # These are MIDI notes that couldn't be matched to SVG note heads
        if svgX == 0 and svgY == 0:
            skipped_no_svg += 1
            continue

        if t not in notes_at_time:
            notes_at_time[t] = []
        # Find end time for this note
        end_t = end_time
        for s, et in note_instances.get(note, []):
            if s == t:
                end_t = et
                break

        # Convert to Blender Y - THIS IS THE KEY TRANSFORMATION
        _, bY = svg_to_blender(svgX, svgY)

        notes_at_time[t].append((
            note,
            svgX,
            svgY,
            bY,  # Blender Y for all logic
            event.get('name', ''),
            end_t
        ))

    if skipped_no_svg > 0:
        print(f"  Skipped {skipped_no_svg} notes without SVG coordinates")

    # Sort notes at each time by bY ASCENDING (lowest bY first = curve1's preferred slot)
    for t in notes_at_time:
        notes_at_time[t].sort(key=lambda x: x[3])  # Sort by bY ascending

    # === STEP 2: Process ALL timestamps, distributing notes to curves ===
    #
    # ALGORITHM: Forward-time processing that respects note durations
    # and maintains Y-ordering.
    #
    # For each curve, track:
    # - Current note (if any)
    # - When current note ends
    # - Current bY position
    #
    # At each timestamp:
    # 1. Check which curves' notes have ended → they need new landings
    # 2. Sort available notes by bY
    # 3. Assign notes to curves maintaining Y-ordering
    #
    # The key constraint: at any time, curve1.bY <= curve2.bY <= ... <= curve7.bY

    # Track state for each curve
    curve_state = {}
    for curve_name in curve_names:
        curve_state[curve_name] = {
            'note': None,
            'end_time': 0.0,
            'bY': None,
            'landings': []
        }

    # Process all timestamps in order
    all_times = sorted(notes_at_time.keys())

    for current_time in all_times:
        notes_starting = notes_at_time.get(current_time, [])
        if not notes_starting:
            continue

        # Sort notes by bY ascending (lowest pitch = lowest bY first)
        notes_sorted = sorted(notes_starting, key=lambda x: x[3])
        notes_by_bY = {n[3]: n for n in notes_sorted}  # bY -> note data

        # Get current bY for each curve (what they're currently at)
        current_bYs = []
        for curve_idx, curve_name in enumerate(curve_names):
            state = curve_state[curve_name]
            if state['bY'] is not None:
                current_bYs.append((curve_idx, state['bY'], state['end_time']))
            else:
                current_bYs.append((curve_idx, None, 0.0))

        # Determine which curves need to land (note ended or never started)
        curves_needing_landing = []
        curves_holding = []  # Curves still on a valid note

        for curve_idx, curve_name in enumerate(curve_names):
            state = curve_state[curve_name]

            if state['note'] is None:
                curves_needing_landing.append(curve_idx)
            elif current_time >= state['end_time'] - 0.01:
                curves_needing_landing.append(curve_idx)
            else:
                # Check if same note restarts
                for note, svgX, svgY, bY, noteName, end_t in notes_sorted:
                    if note == state['note']:
                        state['end_time'] = max(state['end_time'], end_t)
                        break
                curves_holding.append(curve_idx)

        if not curves_needing_landing:
            continue  # All curves are holding - skip this timestamp

        # Build the bY constraints from holding curves
        # curves_holding have fixed bY positions that constrain the needing curves
        holding_bYs = {}
        for curve_idx in curves_holding:
            holding_bYs[curve_idx] = curve_state[curve_names[curve_idx]]['bY']

        # Assign notes to curves that need landing
        # Process from lowest to highest curve index
        used_notes = set()

        for curve_idx in sorted(curves_needing_landing):
            curve_name = curve_names[curve_idx]
            state = curve_state[curve_name]

            # Determine min/max bY for this curve based on neighbors
            min_bY = 0.0
            max_bY = float('inf')

            # Look at all lower-indexed curves (both holding and already assigned this round)
            for other_idx in range(curve_idx):
                other_bY = curve_state[curve_names[other_idx]]['bY']
                if other_bY is not None:
                    min_bY = max(min_bY, other_bY)

            # Look at all higher-indexed holding curves
            for other_idx in range(curve_idx + 1, len(curve_names)):
                if other_idx in holding_bYs:
                    max_bY = min(max_bY, holding_bYs[other_idx])

            # Find the lowest-bY note in range [min_bY, max_bY] that hasn't been used
            best_note = None
            for note, svgX, svgY, bY, noteName, end_t in notes_sorted:
                if id(notes_sorted) in used_notes:
                    continue
                note_key = (note, current_time)
                if note_key in used_notes:
                    continue
                if bY >= min_bY - 0.001 and bY <= max_bY + 0.001:
                    best_note = (note, svgX, svgY, bY, noteName, end_t)
                    used_notes.add(note_key)
                    break

            if best_note is not None:
                note, svgX, svgY, bY, noteName, end_t = best_note
                state['landings'].append((current_time, note, svgX, svgY, bY, noteName))
                state['note'] = note
                state['end_time'] = end_t
                state['bY'] = bY    # Build completed_curves from curve_state
    completed_curves = {}
    for curve_name in curve_names:
        completed_curves[curve_name] = curve_state[curve_name]['landings']

    # === STEP 3: Build final curve data ===
    # Convert back to the output format (keeping both SVG and Blender coords for flexibility)

    curves = {}
    for cn in curve_names:
        points = []
        landings = completed_curves.get(cn, [])

        # Add start point (fly-in) - use first event's position
        first_svgX = first_event.get('svgX', 0)
        first_svgY = first_event.get('svgY', 0)
        _, first_bY = svg_to_blender(first_svgX, first_svgY)

        start_point = {
            'noteName': first_event.get('name', ''),
            'note': first_event.get('note', 0),
            'svgX': first_svgX,
            'svgY': first_svgY,
            'bY': first_bY,
            'timestamp': 0.0,
            'pointType': 'start'
        }
        points.append(start_point)

        # Add landing points
        for t, note, svgX, svgY, bY, noteName in landings:
            if t > 0:
                point = {
                    'noteName': noteName,
                    'note': note,
                    'svgX': svgX,
                    'svgY': svgY,
                    'bY': bY,
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
                'bY': last_point['bY'],
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
