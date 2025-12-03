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

    # curve1_points = json_data.get('curves', {}).get('curve1', {}).get('points', [])
    # if not curve1_points:
    #     print("No data found for 'curves.curve1.points' in JSON.")
    #     return 1

    # curve1_landings = [pt for pt in curve1_points if pt.get('type') == 'landing']
    # if not curve1_landings:
    #     print("No landing points found in 'curve1'.")
    #     return 1

    # print(f"  Found {len(curve1_landings)} landings for curve1.")

    all_trails_collection = bpy.data.collections.get('Energy Trails')
    if all_trails_collection is None:
        print("ERROR: 'Energy Trails' collection not found in Blender file.", file=sys.stderr)
        return 1

    camera_controller = all_trails_collection.objects['Energy Trails']
    if camera_controller is None:
        print("ERROR: 'Energy Trails' object not found in Blender file.", file=sys.stderr)
        return 1

    for trail in all_trails_collection.objects:
        if trail is None:
            print("ERROR: An Energy Trail object not found in Blender file.", file=sys.stderr)
            return 1

        name = trail.name
        print(f"\nProcessing Energy Trail: '{name}'")

    # First, clear existing keyframes on the Energy Trails's X location
    camera_controller.animation_data_clear()

    # Now, for every landing, create an X-value keyframe at time + MUSIC_START_OFFSET_FRAMES
    # Make sure the keyframes use fractional frames and linear interpolation.
    for landing in curve1_landings:
        time_sec = landing.get('timestamp', 0.0)
        x_value = landing.get('x', 0.0)
        frame_num = MUSIC_START_OFFSET_FRAMES + (time_sec * GLOBAL_FPS)
        camera_controller.location.x = x_value
        camera_controller.keyframe_insert(data_path="location", index=0, frame=frame_num)
        print(f"  Inserted keyframe at time {time_sec:.6f} frame {frame_num:.6f} with X={x_value:.6f}")

if __name__ == "__main__":
    main()
