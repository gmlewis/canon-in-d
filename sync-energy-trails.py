#!/usr/bin/env python3
"""
Blender 5.0 script to sync the "Energy Trail" curves from pre-generated data.
Run this script from within Blender's scripting environment.

This script:
1. Reads the JSON data file and creates animation keyframes for the "Energy Trails".
"""

import bpy
import json
import sys
import os

# ============================================================================
# Configuration
# ============================================================================
# Path to the JSON file (relative to the .blend file or absolute)
JSON_FILE_PATH = "note-jumping-curves.json"

GLOBAL_FPS = 60
MUSIC_START_OFFSET_FRAMES = 120  # Frame at which music starts (2 seconds at 60 FPS)
INITIAL_ENERGY_TRAIL_X_OFFSET = -5.0
FIRST_LANDING_POINT_ENERGY_TRAIL_X_OFFSET_AT_T0 = 3.152

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


def load_json_data_file(filepath):
    """Load the JSON structure."""
    data = load_json_data(filepath)
    if data is None:
        return None
    if 'curves' not in data:
        print(f"ERROR: JSON file '{filepath}' is missing required sections.", file=sys.stderr)
        return None
    return data


def main():
    print("=" * 70)
    print("Sync Energy Trails (Blender Script)")
    print("=" * 70)

    print(f"\nLoading JSON data from '{JSON_FILE_PATH}'...")
    json_data = load_json_data_file(JSON_FILE_PATH)
    if json_data is None:
        print("Aborting due to JSON load failure.")
        return 1

    note_jumping_curves_collection = bpy.data.collections.get('Note Jumping Curves')
    if note_jumping_curves_collection is None:
        print("ERROR: 'Note Jumping Curves' collection not found in Blender file.", file=sys.stderr)
        return 1
    curves_parent = note_jumping_curves_collection.objects['Note Jumping Curves Parent']
    if curves_parent is None:
        print("ERROR: 'Note Jumping Curves Parent' object not found in Blender file.", file=sys.stderr)
        return 1

    all_trails_collection = bpy.data.collections.get('Energy Trails')
    if all_trails_collection is None:
        print("ERROR: 'Energy Trails' collection not found in Blender file.", file=sys.stderr)
        return 1

    for trail in all_trails_collection.objects:
        if trail is None:
            print("ERROR: An Energy Trail object not found in Blender file.", file=sys.stderr)
            return 1

        trail_name = trail.name
        print(f"\nProcessing Energy Trail: '{trail_name}'")
        curve_name = trail_name.replace("Energy Trail ", "curve")
        json_curve_points = json_data.get('curves', {}).get(curve_name, {}).get('points', [])
        if not json_curve_points:
            print(f"  No data found for '{curve_name}' in JSON.")
            return 1
        json_curve_landings = [pt for pt in json_curve_points if pt.get('type') == 'landing']
        if not json_curve_landings:
            print(f"No landing points found in '{curve_name}' in JSON.")
            return 1
        print(f"  Found {len(json_curve_landings)} landings for '{curve_name}'.")

        blender_curve = curves_parent.children.get(f"{curve_name}_curve")
        if blender_curve is None:
            print(f"ERROR: Blender curve object '{curve_name}_curve' not found.", file=sys.stderr)
            return 1

        # TODO: Read the first two animation keyframes for the X location of the trail.
        # * If the first keyframe is not at frame 1, warn and skip.
        # * If the first keyframe's X location is set, print this value and use it instead of INITIAL_ENERGY_TRAIL_X_OFFSET.
        # * If the second keyframe is not at frame MUSIC_START_OFFSET_FRAMES, warn and skip.
        # * If the second keyframe's X location is set, print this value and use it instead of FIRST_LANDING_POINT_ENERGY_TRAIL_X_OFFSET_AT_T0.
        # * Clear all keyframes on the trail's X location.
        # * Create the first two keyframes with the determined X locations, one on frame 1 and one on MUSIC_START_OFFSET_FRAMES.
        # * Note that frame MUSIC_START_OFFSET_FRAMES corresponds to time 0 in the JSON data and note that this also represents the first landing for each curve and should be the first landing position from the JSON data.
        # * Now, for every pair of landings, first ask Blender to measure the physical length of the curve between the first and second landing points.
        # * Using this length, add this length to the previous keyframe's X location to get the new X location for the second landing point and make an animation keyframe for this curve's energy trail at the appropriate frame based on the timestamp of the second landing point (using fractional floating-point frames for accuracy).

if __name__ == "__main__":
    main()
