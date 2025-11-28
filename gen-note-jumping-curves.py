#!/usr/bin/env python3
"""
Blender 5.0 script to generate note jumping curves from a JSON file.
Run this script from within Blender's scripting environment.
"""

import bpy
import json
import sys

# Configuration
JSON_FILE = "CanonInD.json"
COLLECTION_NAME = "Note Jumping Curves"
PARENT_NAME = "Note Jumping Curves Parent"


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

    The JSON structure is a list of tracks, where each track is a list of events.
    Each event has 'type', 'time', and other fields.
    """
    # Collect all noteOn and noteOff events with their times
    events = []

    for track in data:
        # Each track is a list of events directly
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
                events.append((time, 1, event))  # 1 = note starts
            elif event_type == 'noteOff':
                events.append((time, -1, event))  # -1 = note ends

    # Sort by time, with noteOff before noteOn at same time to avoid overcounting
    # (if a note ends and another starts at the same instant)
    events.sort(key=lambda x: (x[0], x[1]))

    # Sweep through to find maximum concurrent notes
    current_count = 0
    max_count = 0

    for time, delta, event in events:
        current_count += delta
        if current_count > max_count:
            max_count = current_count

    return max_count


def collect_note_events(data):
    """
    Collect all noteOn and noteOff events from the JSON data.
    Returns a list of (time, event_type, event) tuples sorted by time.
    event_type is 'noteOn' or 'noteOff'.
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
    # This ensures notes are released before new ones are grabbed
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'noteOff' else 1))

    return events


def build_curve_data(data, max_curves):
    """
    Build the curve data dictionary based on note events.

    Algorithm:
    1. All curves start at time -1 (before song)
    2. At each noteOn, assign the lowest available curve to the lowest pitch
    3. At each noteOff, free up the curve that was playing that note
    4. Free curves "merge" to their nearest active neighbor
    5. All curves end at (last_note_time + 1)

    Returns a dict: {'curve1': [...], 'curve2': [...], ...}
    Each curve's list contains dicts: {noteName, svgX, svgY, timestamp}
    """
    events = collect_note_events(data)

    if not events:
        return {}

    # Find the end time of the song (last noteOff time)
    end_time = max(e[0] for e in events) + 1.0

    # Initialize curve data structure
    curves = {f'curve{i+1}': [] for i in range(max_curves)}

    # Track which curves are currently "busy" (playing a note)
    # Maps curve_name -> (note_number, noteOn_event) or None if free
    curve_assignments = {f'curve{i+1}': None for i in range(max_curves)}

    # Track currently playing notes: note_number -> noteOn_event
    active_notes = {}

    # For the starting point at time -1, we need to know the first note
    # Find the first noteOn event
    first_note_event = None
    for time, event_type, event in events:
        if event_type == 'noteOn':
            first_note_event = event
            break

    if first_note_event is None:
        return curves

    # All curves start at time -1, jumping to the first note position
    start_point = {
        'noteName': first_note_event.get('name', ''),
        'svgX': first_note_event.get('svgX', 0),
        'svgY': first_note_event.get('svgY', 0),
        'timestamp': -1.0
    }
    for curve_name in curves:
        curves[curve_name].append(start_point.copy())

    # Process events in time order
    # We need to handle events at the same timestamp together
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

        # Process noteOff events first - free up curves
        for time, event_type, event in note_offs:
            note_num = event.get('note')
            if note_num in active_notes:
                del active_notes[note_num]

            # Find which curve was playing this note and free it
            for curve_name, assignment in curve_assignments.items():
                if assignment is not None and assignment[0] == note_num:
                    curve_assignments[curve_name] = None
                    break

        # Process noteOn events - assign curves to notes
        for time, event_type, event in note_ons:
            note_num = event.get('note')
            active_notes[note_num] = event

        # Now reassign all curves based on current active notes
        # Sort active notes by pitch (note number)
        sorted_active = sorted(active_notes.items(), key=lambda x: x[0])

        # Get list of all curves sorted by number
        curve_names = [f'curve{i+1}' for i in range(max_curves)]

        # Assign curves to active notes (lowest curve to lowest pitch)
        # First, mark all curves as free
        new_assignments = {cn: None for cn in curve_names}

        for idx, (note_num, note_event) in enumerate(sorted_active):
            if idx < max_curves:
                curve_name = curve_names[idx]
                new_assignments[curve_name] = (note_num, note_event)

        # For free curves, they should merge to their nearest active neighbor
        # We need to determine where each curve should jump to at this timestamp
        for curve_idx, curve_name in enumerate(curve_names):
            if new_assignments[curve_name] is not None:
                # This curve is assigned to an active note
                note_num, note_event = new_assignments[curve_name]
                point = {
                    'noteName': note_event.get('name', ''),
                    'svgX': note_event.get('svgX', 0),
                    'svgY': note_event.get('svgY', 0),
                    'timestamp': current_time
                }
                curves[curve_name].append(point)
            else:
                # This curve is free - merge to nearest active neighbor
                # Find nearest active curve (by curve index)
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
                        'svgX': nearest_event.get('svgX', 0),
                        'svgY': nearest_event.get('svgY', 0),
                        'timestamp': current_time
                    }
                    curves[curve_name].append(point)
                # If no active notes at all (shouldn't happen mid-song), skip

        # Update assignments for next iteration
        curve_assignments = new_assignments

    # All curves end at end_time, jumping off to the right
    # Use the last active note's position but with end_time timestamp
    # Since we're jumping "off to the right", we'll use the last position
    # of each curve but extrapolate X position
    for curve_name in curves:
        if curves[curve_name]:
            last_point = curves[curve_name][-1]
            # Create end point - keep same Y, extend X to the right
            end_point = {
                'noteName': last_point['noteName'],
                'svgX': last_point['svgX'] + 500,  # Offset to the right
                'svgY': last_point['svgY'],
                'timestamp': end_time
            }
            curves[curve_name].append(end_point)

    return curves


def deduplicate_curve_points(curves):
    """
    Remove consecutive duplicate points in each curve.
    A point is duplicate if it has the same svgX and svgY as the previous point.
    """
    for curve_name, points in curves.items():
        if len(points) <= 1:
            continue

        deduped = [points[0]]
        for point in points[1:]:
            last = deduped[-1]
            # Keep point if position changed (allowing for floating point comparison)
            if (abs(point['svgX'] - last['svgX']) > 0.001 or
                abs(point['svgY'] - last['svgY']) > 0.001):
                deduped.append(point)

        curves[curve_name] = deduped

    return curves


def setup_collection():
    """
    Set up the 'Note Jumping Curves' collection.
    - If it exists, delete all objects within it.
    - If it doesn't exist, create it.
    Returns the collection.
    """
    scene = bpy.context.scene

    # Check if collection already exists
    collection = bpy.data.collections.get(COLLECTION_NAME)

    if collection is not None:
        # Collection exists - delete all objects within it
        print(f"Collection '{COLLECTION_NAME}' exists. Removing all objects...")

        # We need to remove objects from the collection
        # First, unlink and delete all objects
        objects_to_delete = list(collection.objects)
        for obj in objects_to_delete:
            bpy.data.objects.remove(obj, do_unlink=True)

        print(f"Removed {len(objects_to_delete)} objects from collection.")
    else:
        # Collection doesn't exist - create it
        print(f"Creating new collection '{COLLECTION_NAME}'...")
        collection = bpy.data.collections.new(COLLECTION_NAME)
        scene.collection.children.link(collection)

    return collection


def create_parent_empty(collection):
    """
    Create an empty object to serve as parent for all curves.
    Returns the empty object.
    """
    # Create an empty object
    empty = bpy.data.objects.new(PARENT_NAME, None)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.empty_display_size = 1.0

    # Link it to the collection
    collection.objects.link(empty)

    print(f"Created parent empty '{PARENT_NAME}'")

    return empty


def main():
    print("=" * 60)
    print("Note Jumping Curves Generator")
    print("=" * 60)

    # Step 1: Load JSON data
    print(f"\nLoading '{JSON_FILE}'...")
    data = load_json_data(JSON_FILE)

    if data is None:
        print("Aborting due to JSON load failure.")
        return

    print(f"Successfully loaded JSON with {len(data)} tracks.")

    # Step 2: Compute MAX_CURVES (maximum concurrent notes)
    print("\nAnalyzing note overlaps...")
    MAX_CURVES = compute_max_concurrent_notes(data)
    print(f"MAX_CURVES = {MAX_CURVES} (maximum concurrent notes)")

    # Step 3: Build curve data
    print("\nBuilding curve data...")
    curves = build_curve_data(data, MAX_CURVES)
    curves = deduplicate_curve_points(curves)

    # Print summary of curve data
    print(f"\nCurve data summary:")
    for curve_name in sorted(curves.keys(), key=lambda x: int(x.replace('curve', ''))):
        points = curves[curve_name]
        print(f"  {curve_name}: {len(points)} points")
        if points:
            print(f"    First: t={points[0]['timestamp']:.2f}, note={points[0]['noteName']}")
            print(f"    Last:  t={points[-1]['timestamp']:.2f}, note={points[-1]['noteName']}")

    # Step 4 & 5: Set up collection (create or clean existing)
    print(f"\nSetting up collection '{COLLECTION_NAME}'...")
    collection = setup_collection()

    # Step 6: Create parent empty
    print("\nCreating parent empty node...")
    parent_empty = create_parent_empty(collection)

    print("\n" + "=" * 60)
    print("Stage 2 complete!")
    print(f"  - MAX_CURVES: {MAX_CURVES}")
    print(f"  - Collection: '{COLLECTION_NAME}'")
    print(f"  - Parent: '{PARENT_NAME}'")
    print(f"  - Curves built: {len(curves)}")
    print("=" * 60)

    return curves, collection, parent_empty


if __name__ == "__main__":
    main()
