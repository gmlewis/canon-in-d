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
import os
import re
import sys
from collections import defaultdict, deque

# ============================================================================
# Configuration - These values are written to the output JSON for Blender
# ============================================================================
INPUT_JSON_FILE = "CanonInD.json"
OUTPUT_JSON_FILE = "note-jumping-curves.json"
SVG_FILE = "Canon_in_D-single-svg-printing_NoteHeads.svg"
NOTEHEAD_MAP_FILE = "notehead-map.json"

# Blender scene configuration
COLLECTION_NAME = "Note Jumping Curves"
PARENT_NAME = "Note Jumping Curves Parent"
MAX_JUMPING_CURVE_Z_OFFSET = 0.5  # Height of the arc peak between notes

# User's scale factor applied AFTER SVG import in Blender
# Set this to match whatever scale you apply to the imported SVG in Blender
USER_SCALE_FACTOR = 100.0  # You scale the SVG by 100x after import

# These will be computed from SVG file / notehead map metadata
SVG_SCALE = None       # Computed from notehead map metadata
X_SCALE = None         # Same as SVG_SCALE (uniform scaling)
Y_SCALE = None         # Same as SVG_SCALE (positive, for use in formula)
VIEWBOX_HEIGHT = None  # Height of SVG viewBox (for Y flip calculation)


def svg_to_blender(svgX, svgY):
    """Transform SVG coordinates to Blender coordinates."""
    blenderX = svgX * X_SCALE
    blenderY = (VIEWBOX_HEIGHT - svgY) * Y_SCALE
    return blenderX, blenderY


def blender_to_svg(blenderX, blenderY):
    """Transform Blender coordinates back to SVG coordinates."""
    svgX = blenderX / X_SCALE
    svgY = VIEWBOX_HEIGHT - (blenderY / Y_SCALE)
    return svgX, svgY


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


def format_note_on_key(time_value, midi_note, precision=6):
    """Create the canonical index key used by notehead-map.json."""
    if time_value is None or midi_note is None:
        return None
    rounded_time = round(time_value + 1e-12, precision)
    return f"{rounded_time:.{precision}f}|{int(midi_note)}"


def load_notehead_map_data(filepath):
    """Load the canonical notehead map JSON structure."""
    data = load_json_data(filepath)
    if data is None:
        return None
    if 'heads' not in data or 'metadata' not in data:
        print(f"ERROR: Notehead map '{filepath}' is missing required sections.", file=sys.stderr)
        return None
    return data


def build_notehead_lookup(notehead_map):
    """Build a lookup of canonical heads keyed by (time|note)."""
    heads = notehead_map.get('heads', [])
    index = notehead_map.get('index', {})
    by_note_on = index.get('byNoteOn', {})

    id_lookup = {}
    for idx, head in enumerate(heads):
        head_id = head.get('svgId') or head.get('id')
        if not head_id:
            head_id = f"head_{idx}"
            head['svgId'] = head_id
        id_lookup[head_id] = head

    lookup = {}
    for key, id_list in by_note_on.items():
        queue = deque()
        for head_id in id_list:
            head = id_lookup.get(head_id)
            if head:
                queue.append(head)
        if queue:
            lookup[key] = queue
    return lookup


def apply_notehead_map_to_events(data, lookup):
    """Attach SVG coordinates to noteOn events using the canonical map."""
    total = 0
    matched = 0
    missing = []

    for track in data:
        if not isinstance(track, list):
            continue
        for event in track:
            if not isinstance(event, dict) or event.get('type') != 'noteOn':
                continue
            total += 1
            key = format_note_on_key(event.get('time'), event.get('note'))
            if key is None:
                missing.append((event.get('time'), event.get('note')))
                continue
            bucket = lookup.get(key)
            if not bucket:
                missing.append((event.get('time'), event.get('note')))
                continue
            head = bucket.popleft()
            event['svgX'] = head.get('svgX', 0.0)
            event['svgY'] = head.get('svgY', 0.0)
            event['svgId'] = head.get('svgId') or head.get('id')
            if head.get('noteName'):
                event['name'] = head['noteName']
            matched += 1

    return matched, total, missing


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
    all_note_keys = set()
    note_usage_counts = defaultdict(int)
    note_lookup = {}
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

        note_key = (round(t, 5), note)
        all_note_keys.add(note_key)
        note_lookup[note_key] = {
            'time': t,
            'noteName': event.get('name', ''),
            'note': note,
            'svgX': svgX,
            'svgY': svgY,
            'bY': bY,
            'end_t': end_t
        }

    if skipped_no_svg > 0:
        print(f"  Skipped {skipped_no_svg} notes without SVG coordinates")

    # Sort notes at each time by bY ASCENDING (lowest bY first = curve1's preferred slot)
    for t in notes_at_time:
        notes_at_time[t].sort(key=lambda x: x[3])  # Sort by bY ascending

    # === STEP 2: Trace curves forward in time (bottom-up) ===
    # Each curve walks the timeline independently while staying above
    # all previously completed (lower-numbered) curves.

    completed_curves = {}
    notes_covered = set()
    times_to_process = sorted(note_on_times)

    RANK_WEIGHT = 12.0
    MOVE_WEIGHT = 1.0
    USAGE_WEIGHT = 12.0
    COVERED_PENALTY = 25.0
    ACTIVE_WEIGHT = 40.0

    active_note_counts = defaultdict(int)

    for curve_idx, curve_name in enumerate(curve_names):
        curve_landings = []

        current_note = None
        current_note_key = None
        current_note_end = 0.0
        current_bY = None
        last_landing_time = 0.0

        for current_time in times_to_process:
            notes_available = notes_at_time.get(current_time, [])
            if not notes_available:
                continue

            # Release the current note if it has ended
            if (current_note is not None and
                    current_time >= current_note_end - 0.01):
                if current_note_key is not None and active_note_counts[current_note_key] > 0:
                    active_note_counts[current_note_key] -= 1
                current_note = None
                current_note_key = None

            # Extend current note if it restarts at this timestamp
            if current_note is not None and current_time < current_note_end - 0.01:
                for note, svgX, svgY, bY, noteName, end_t in notes_available:
                    if note == current_note and abs(bY - current_bY) < 0.01:
                        current_note_end = max(current_note_end, end_t)
                        break
                continue

            best_candidate = None
            best_score = float('inf')
            target_rank = min(curve_idx, len(notes_available) - 1)

            for rank, (note, svgX, svgY, bY, noteName, end_t) in enumerate(notes_available):
                note_key = (round(current_time, 5), note)

                origin_bY = current_bY if current_bY is not None else bY
                origin_time = last_landing_time if curve_landings else current_time

                if not is_position_valid(curve_idx + 1, origin_bY, origin_time,
                                         bY, current_time, completed_curves):
                    continue

                rank_score = abs(rank - target_rank)
                move_score = abs(bY - current_bY) if current_bY is not None else 0.0
                usage_score = note_usage_counts[note_key]
                active_penalty = active_note_counts[note_key] * ACTIVE_WEIGHT
                covered_penalty = COVERED_PENALTY if note_key in notes_covered else 0.0
                total_score = (rank_score * RANK_WEIGHT) + \
                              (move_score * MOVE_WEIGHT) + \
                              (usage_score * USAGE_WEIGHT) + \
                              active_penalty + \
                              covered_penalty

                if total_score < best_score:
                    best_score = total_score
                    best_candidate = (note, svgX, svgY, bY, noteName, end_t, note_key)

            if best_candidate is None:
                continue

            note, svgX, svgY, bY, noteName, end_t, note_key = best_candidate
            curve_landings.append((current_time, note, svgX, svgY, bY, noteName))
            current_note = note
            current_note_key = note_key
            current_note_end = end_t
            current_bY = bY
            last_landing_time = current_time
            note_usage_counts[note_key] += 1
            active_note_counts[note_key] += 1
            notes_covered.add(note_key)

        completed_curves[curve_name] = curve_landings

    missing_set = all_note_keys - notes_covered
    missing_notes = len(missing_set)

    def insert_landing(curve_landings, info):
        entry = (info['time'], info['note'], info['svgX'], info['svgY'], info['bY'], info['noteName'])
        inserted = False
        for idx, (landing_time, *_rest) in enumerate(curve_landings):
            if landing_time > info['time']:
                curve_landings.insert(idx, entry)
                inserted = True
                break
        if not inserted:
            curve_landings.append(entry)

    FINAL_TIME_THRESHOLD = 241.7

    FINAL_CHORD_REQUIREMENTS = [
        {'curve': 'curve1', 'note': 38, 'time': 241.79834580873745},  # D2
        {'curve': 'curve2', 'note': 45, 'time': 241.79834580873745},  # A2
        {'curve': 'curve3', 'note': 54, 'time': 241.79834580873745},  # F#3
        {'curve': 'curve4', 'note': 62, 'time': 241.79834580873745},  # D4
        {'curve': 'curve5', 'note': 66, 'time': 241.81834497553882},  # F#4
        {'curve': 'curve6', 'note': 69, 'time': 241.8408461381869},   # A4
        {'curve': 'curve7', 'note': 74, 'time': 241.86084530498826},  # D5
    ]

    def resolve_chord_entries(requirements, chord_label):
        entries = []
        for requirement in requirements:
            note_key = (round(requirement['time'], 5), requirement['note'])
            info = note_lookup.get(note_key)
            if not info:
                print(
                    f"  WARNING: Missing note info for {chord_label} note={requirement['note']} "
                    f"time={requirement['time']:.5f}"
                )
                continue

            entries.append({
                'curve': requirement['curve'],
                'time': info['time'],
                'note': info['note'],
                'noteName': info['noteName'],
                'svgX': info['svgX'],
                'svgY': info['svgY'],
                'bY': info['bY']
            })
        return entries

    def apply_final_chord_layout():
        for curve_name in curve_names:
            landings = completed_curves.get(curve_name, [])
            completed_curves[curve_name] = [
                landing for landing in landings if landing[0] < FINAL_TIME_THRESHOLD
            ]

        for entry in resolve_chord_entries(FINAL_CHORD_REQUIREMENTS, "final chord"):
            insert_landing(completed_curves.setdefault(entry['curve'], []), entry)
            print(f"  INFO: Final chord assigned {entry['noteName']} to {entry['curve']}")
    apply_final_chord_layout()

    # Rebuild coverage to reflect the enforced chord
    notes_covered = set()
    for curve_name in curve_names:
        for landing in completed_curves.get(curve_name, []):
            notes_covered.add((round(landing[0], 5), landing[1]))

    def can_assign_missing(curve_idx, info):
        """Check whether a missed landing can be inserted into a curve."""
        bY = info['bY']
        t = info['time']

        curve_landings = completed_curves.get(curve_names[curve_idx], [])

        prev_landing = None
        for landing in curve_landings:
            if abs(landing[0] - t) < 1e-6:
                return False  # landing already exists at this timestamp
            if landing[0] < t:
                prev_landing = landing
            else:
                break

        if prev_landing is not None:
            prev_key = (round(prev_landing[0], 5), prev_landing[1])
            prev_end = note_lookup.get(prev_key, {}).get('end_t')
            if prev_end is not None and prev_end - 0.05 > t:
                return False  # curve is still sustaining a previous note

        # Lower curves must stay <= this curve's Y
        for lower_idx in range(curve_idx):
            lower_curve = completed_curves.get(curve_names[lower_idx], [])
            lower_bY = get_curve_bY_at_time(lower_curve, t)
            if lower_bY is not None and bY < lower_bY - 0.001:
                return False

        # Higher curves must stay >= this curve's Y
        for higher_idx in range(curve_idx + 1, len(curve_names)):
            higher_curve = completed_curves.get(curve_names[higher_idx], [])
            higher_bY = get_curve_bY_at_time(higher_curve, t)
            if higher_bY is not None and bY > higher_bY + 0.001:
                return False

        return True

    def backfill_missing_notes(missing_keys):
        inserted = 0
        ordered_keys = sorted(
            [k for k in missing_keys if note_lookup.get(k, {}).get('time', 0.0) < FINAL_TIME_THRESHOLD],
            key=lambda k: note_lookup.get(k, {}).get('time', 0.0)
        )
        for key in ordered_keys:
            info = note_lookup.get(key)
            if not info:
                continue
            for curve_idx in range(len(curve_names)):
                if can_assign_missing(curve_idx, info):
                    insert_landing(completed_curves.setdefault(curve_names[curve_idx], []), info)
                    notes_covered.add(key)
                    inserted += 1
                    break
        return inserted

    if missing_notes > 0:
        recovered = backfill_missing_notes(missing_set)
        if recovered:
            print(f"  INFO: Backfilled {recovered} missing notes via post-processing")
        missing_set = all_note_keys - notes_covered
        missing_notes = len(missing_set)

    if missing_notes > 0:
        coverage_percent = (len(notes_covered) / len(all_note_keys)) * 100 if all_note_keys else 0
        print(f"  WARNING: {missing_notes} matched notes not covered by any curve "
              f"({coverage_percent:.1f}% coverage)")
        sample = list(missing_set)[:5]
        for key in sample:
            info = note_lookup.get(key)
            if info:
                print(f"    Missing note {info['noteName']} (MIDI {info['note']}) at t={info['time']:.3f}s")

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

        # Use the FINAL landing we just emitted to anchor the fly-out segment.
        last_landing_point = None
        for candidate in reversed(bezier_points):
            if candidate.get('type') == 'landing':
                last_landing_point = candidate
                break

        if last_landing_point is None:
            bezier_curves[curve_name] = {
                'landingCount': len(landing_points),
                'bezierPointCount': len(bezier_points),
                'points': bezier_points
            }
            continue

        last_x = last_landing_point['x']
        last_y = last_landing_point['y']
        last_svg_x = last_landing_point['svgX']
        last_svg_y = last_landing_point['svgY']
        last_note_name = last_landing_point.get('noteName', '')
        last_timestamp = last_landing_point.get('timestamp', 0.0)

        # Add arc peak between last landing and fly-off
        fly_off_peak = {
            'x': last_x + (FLY_OFFSET / 2.0),
            'y': last_y,
            'z': z_offset,
            'type': 'peak',
            'noteName': f"{last_note_name} -> fly-off peak",
            'note': None,
            'timestamp': last_timestamp + 0.5,
            'pointType': 'fly_off_peak',
            'svgX': last_svg_x + (FLY_OFFSET / 2.0 / X_SCALE),
            'svgY': last_svg_y
        }
        bezier_points.append(fly_off_peak)

        # Add fly-off end point (hovering at z_offset, WAY off to the right)
        fly_off_end = {
            'x': last_x + 5.0,  # 5 meters past last landing - way off-screen
            'y': last_y,
            'z': z_offset,  # Hovering, not on the ground
            'type': 'fly_off',
            'noteName': f"{last_note_name} -> fly-off",
            'note': None,
            'timestamp': last_timestamp + 1.0,
            'pointType': 'fly_off_end',
            'svgX': last_svg_x + 500,  # Way off-screen in SVG coords
            'svgY': last_svg_y
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

    # Step 0: Load canonical note head mapping for scale + coordinates
    print(f"\nLoading note head map '{NOTEHEAD_MAP_FILE}'...")
    notehead_map = load_notehead_map_data(NOTEHEAD_MAP_FILE)
    if notehead_map is None:
        print("Aborting due to note head map load failure.")
        return 1

    metadata = notehead_map.get('metadata', {})
    scale = metadata.get('svgScale')
    viewbox_height = metadata.get('viewboxHeight')
    if scale is None or viewbox_height is None:
        print("ERROR: Note head map is missing scale metadata. Aborting.")
        return 1

    X_SCALE = scale
    Y_SCALE = scale
    VIEWBOX_HEIGHT = viewbox_height
    SVG_SCALE = scale

    source_svg = metadata.get('sourceSvg', SVG_FILE)
    total_heads = metadata.get('totalHeads', 'unknown')
    playable_heads = metadata.get('playableHeads', 'unknown')
    tie_only_heads = metadata.get('tieOnlyHeads', 'unknown')
    extra_heads = metadata.get('extraHeads', 'unknown')

    print(f"  Source SVG: {source_svg}")
    print(f"  Scale = {scale:.10f}")
    print(f"  viewBox height = {viewbox_height} (for Y flip)")
    print(f"  Heads: total={total_heads}, playable={playable_heads}, tie-only={tie_only_heads}, extras={extra_heads}")

    lookup = build_notehead_lookup(notehead_map)
    if not lookup:
        print("ERROR: Built empty lookup from note head map. Aborting.")
        return 1

    indexed_count = sum(len(queue) for queue in lookup.values())
    print(f"  Indexed {indexed_count} playable note heads from canonical map")

    # Step 1: Load JSON data
    print(f"\nLoading '{INPUT_JSON_FILE}'...")
    data = load_json_data(INPUT_JSON_FILE)

    if data is None:
        print("Aborting due to JSON load failure.")
        return 1

    print(f"Successfully loaded JSON with {len(data)} tracks.")

    # Step 1b: Assign SVG coordinates using canonical mapping
    print("\nApplying canonical note head mapping...")
    matched_count, total_events, missing_events = apply_notehead_map_to_events(data, lookup)
    print(f"  Matched {matched_count} noteOn events out of {total_events}")
    if missing_events:
        print(f"  WARNING: {len(missing_events)} noteOn events did not resolve to SVG heads")
        for time, note in missing_events[:5]:
            print(f"    Missing note {note} at t={time:.6f}s")
        print("  HINT: Regenerate notehead-map.json if MIDI data changed.")

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
            'svgFile': source_svg,
            'noteheadMap': NOTEHEAD_MAP_FILE,
            'noteheadMapGeneratedAt': metadata.get('generatedAt'),
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
