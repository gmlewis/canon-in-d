#!/usr/bin/env python3
"""
Standalone script to generate note jumping curve data from CanonInD.json.
Outputs a JSON file that can be read by create-curves-from-json.py in Blender.

This script runs OUTSIDE of Blender and does all the data processing.
"""

import json
import sys
import os

# ============================================================================
# Configuration - These values are written to the output JSON for Blender
# ============================================================================
INPUT_JSON_FILE = "CanonInD.json"
OUTPUT_JSON_FILE = "note-jumping-curves.json"

# Blender scene configuration
COLLECTION_NAME = "Note Jumping Curves"
PARENT_NAME = "Note Jumping Curves Parent"
MAX_JUMPING_CURVE_Z_OFFSET = 0.5  # Height of the arc peak between notes

# Scale factors to convert SVG coordinates to Blender units (meters)
# SVG coordinates are in pixels, Blender uses meters by default
# NOTE: Y_SCALE is NEGATIVE to match Blender's SVG import behavior
# (SVG has origin at top-left with Y increasing downward,
#  Blender has Y increasing upward, so we negate Y)
X_SCALE = 0.01   # 1 SVG pixel = 0.01 meters (1 cm)
Y_SCALE = -0.01  # NEGATIVE to flip Y axis like Blender's SVG import

# Offset to align with Blender's SVG import origin
# These values can be tweaked to match the first note's position in the imported SVG
# The first note in the SVG should appear at (FIRST_SVG_NOTE_X_OFFSET, FIRST_SVG_NOTE_Y_OFFSET, 0)
# Blender's SVG import uses this formula: final_y = -svgY * scale + Y_OFFSET
# To calculate Y_OFFSET: Y_OFFSET = first_note_blender_y - (-first_note_svg_y * Y_SCALE)
#                        Y_OFFSET = 0.058835 - (-935.93 * -0.01) = 0.058835 - (-9.3593) = 0.058835 + 9.3593
FIRST_SVG_NOTE_X_OFFSET = 0.035085  # X position of first note in Blender after SVG import
FIRST_SVG_NOTE_Y_OFFSET = 0.058835  # Y position of first note in Blender after SVG import

# First note SVG coordinates (from CanonInD.json) - used to calculate offsets
FIRST_NOTE_SVG_X = 497.25
FIRST_NOTE_SVG_Y = 935.93

# Calculate the offsets to apply to all coordinates
# offset = target_position - (svg_coord * scale)
X_OFFSET = FIRST_SVG_NOTE_X_OFFSET - (FIRST_NOTE_SVG_X * X_SCALE)
Y_OFFSET = FIRST_SVG_NOTE_Y_OFFSET - (FIRST_NOTE_SVG_Y * Y_SCALE)

# How much to offset the final X position (for the "fly off" at end of song)
# This is in SVG units, will be scaled by X_SCALE
END_X_OFFSET = 500


def svg_to_blender_x(svg_x):
    """Convert SVG X coordinate to Blender X coordinate."""
    return svg_x * X_SCALE + X_OFFSET


def svg_to_blender_y(svg_y):
    """Convert SVG Y coordinate to Blender Y coordinate."""
    return svg_y * Y_SCALE + Y_OFFSET


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


def build_curve_data(data, max_curves):
    """
    Build the curve data dictionary based on note events.

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

    # Track which curves are currently "busy" (playing a note)
    curve_assignments = {f'curve{i+1}': None for i in range(max_curves)}

    # Track currently playing notes: note_number -> noteOn_event
    active_notes = {}

    # Find the first noteOn event
    first_note_event = None
    for time, event_type, event in events:
        if event_type == 'noteOn':
            first_note_event = event
            break

    if first_note_event is None:
        return curves, start_time, end_time

    # All curves start at time -1, jumping to the first note position
    start_point = {
        'noteName': first_note_event.get('name', ''),
        'note': first_note_event.get('note', 0),
        'svgX': first_note_event.get('svgX', 0),
        'svgY': first_note_event.get('svgY', 0),
        'timestamp': -1.0,
        'pointType': 'start'
    }
    for curve_name in curves:
        curves[curve_name].append(start_point.copy())

    # Find the time of the last noteOn event
    # After this time, we should NOT reassign curves - just let them stay put
    last_note_on_time = max(t for t, et, e in events if et == 'noteOn')

    # Process events in time order
    i = 0
    while i < len(events):
        current_time = events[i][0]

        # Collect all events at this timestamp
        events_at_time = []
        while i < len(events) and events[i][0] == current_time:
            events_at_time.append(events[i])
            i += 1

        # Separate noteOff and noteOn events
        note_offs = [(t, et, e) for t, et, e in events_at_time if et == 'noteOff']
        note_ons = [(t, et, e) for t, et, e in events_at_time if et == 'noteOn']

        # If we're past the last noteOn, skip processing noteOffs
        # (curves should stay at their final positions)
        if current_time > last_note_on_time:
            continue

        # Process noteOff events first - free up curves
        for time, event_type, event in note_offs:
            note_num = event.get('note')
            if note_num in active_notes:
                del active_notes[note_num]

            for curve_name, assignment in curve_assignments.items():
                if assignment is not None and assignment[0] == note_num:
                    curve_assignments[curve_name] = None
                    break

        # Process noteOn events - assign curves to notes
        for time, event_type, event in note_ons:
            note_num = event.get('note')
            active_notes[note_num] = event

        # Reassign all curves based on current active notes
        sorted_active = sorted(active_notes.items(), key=lambda x: x[0])
        curve_names = [f'curve{i+1}' for i in range(max_curves)]
        new_assignments = {cn: None for cn in curve_names}

        for idx, (note_num, note_event) in enumerate(sorted_active):
            if idx < max_curves:
                curve_name = curve_names[idx]
                new_assignments[curve_name] = (note_num, note_event)

        # Determine where each curve should jump to at this timestamp
        for curve_idx, curve_name in enumerate(curve_names):
            if new_assignments[curve_name] is not None:
                note_num, note_event = new_assignments[curve_name]
                point = {
                    'noteName': note_event.get('name', ''),
                    'note': note_event.get('note', 0),
                    'svgX': note_event.get('svgX', 0),
                    'svgY': note_event.get('svgY', 0),
                    'timestamp': current_time,
                    'pointType': 'landing'
                }
                curves[curve_name].append(point)
            else:
                # This curve is free - merge to nearest active neighbor
                nearest_event = None
                min_distance = float('inf')

                for other_idx, other_name in enumerate(curve_names):
                    if new_assignments[other_name] is not None:
                        distance = abs(other_idx - curve_idx)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_event = new_assignments[other_name][1]

                if nearest_event is not None:
                    point = {
                        'noteName': nearest_event.get('name', ''),
                        'note': nearest_event.get('note', 0),
                        'svgX': nearest_event.get('svgX', 0),
                        'svgY': nearest_event.get('svgY', 0),
                        'timestamp': current_time,
                        'pointType': 'merged'
                    }
                    curves[curve_name].append(point)

        curve_assignments = new_assignments

    # All curves end at end_time, jumping off to the right
    # Use each curve's LAST LANDING (not merged) position for the end point
    for curve_name in curves:
        if curves[curve_name]:
            # Find the last point that was a real "landing" (not merged)
            last_landing = None
            for point in reversed(curves[curve_name]):
                if point['pointType'] in ('landing', 'start'):
                    last_landing = point
                    break

            # If no landing found, fall back to last point
            if last_landing is None:
                last_landing = curves[curve_name][-1]

            end_point = {
                'noteName': last_landing['noteName'],
                'note': last_landing['note'],
                'svgX': last_landing['svgX'] + END_X_OFFSET,
                'svgY': last_landing['svgY'],  # Use last LANDING's Y, not merged Y
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
    print("=" * 70)
    print("Note Jumping Curves Generator (Standalone)")
    print("=" * 70)

    # Step 1: Load JSON data
    print(f"\nLoading '{INPUT_JSON_FILE}'...")
    data = load_json_data(INPUT_JSON_FILE)

    if data is None:
        print("Aborting due to JSON load failure.")
        return 1

    print(f"Successfully loaded JSON with {len(data)} tracks.")

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
            'xScale': X_SCALE,
            'yScale': Y_SCALE,
            'xOffset': X_OFFSET,
            'yOffset': Y_OFFSET,
            'firstSvgNoteXOffset': FIRST_SVG_NOTE_X_OFFSET,
            'firstSvgNoteYOffset': FIRST_SVG_NOTE_Y_OFFSET,
            'curveResolution': 12,
            'handleType': 'AUTO'
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
