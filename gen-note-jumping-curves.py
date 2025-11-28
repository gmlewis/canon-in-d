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

    # Step 3 & 4: Set up collection (create or clean existing)
    print(f"\nSetting up collection '{COLLECTION_NAME}'...")
    collection = setup_collection()

    # Step 5: Create parent empty
    print("\nCreating parent empty node...")
    parent_empty = create_parent_empty(collection)

    print("\n" + "=" * 60)
    print("Stage 1 complete!")
    print(f"  - MAX_CURVES: {MAX_CURVES}")
    print(f"  - Collection: '{COLLECTION_NAME}'")
    print(f"  - Parent: '{PARENT_NAME}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
