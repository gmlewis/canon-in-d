#!/usr/bin/env python3
"""
Blender 5.0 script to sync the "Camera Controller" empty node from pre-generated data.
Run this script from within Blender's scripting environment.

This script:
1. Reads the JSON data file and creates animation keyframes for the "Camera Controller".
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
    """Load the canonical notehead map JSON structure."""
    data = load_json_data(filepath)
    if data is None:
        return None
    if 'curves' not in data:
        print(f"ERROR: JSON file '{filepath}' is missing required sections.", file=sys.stderr)
        return None
    return data


def main():
    print("=" * 70)
    print("Sync Camera Controller (Blender Script)")
    print("=" * 70)

    print(f"\nLoading JSON data from '{JSON_FILE_PATH}'...")
    json_data = load_json_data_file(JSON_FILE_PATH)
    if json_data is None:
        print("Aborting due to JSON load failure.")
        return 1

    curve1_points = json_data.get('curves', {}).get('curve1', {}).get('points', [])
    if not curve1_points:
        print("No data found for 'curves.curve1.points' in JSON.")
        return 1

    curve1_landings = [pt for pt in curve1_points if pt.get('type') == 'landing']
    if not curve1_landings:
        print("No landing points found in 'curve1'.")
        return 1

    print(f"  Found {len(curve1_landings)} landings for curve1.")

    scene = bpy.data.collections.get('Scene')
    if scene is None:
        print("ERROR: 'Scene' collection not found in Blender file.", file=sys.stderr)
        return 1

    camera_controller = scene.objects['Camera Controller']
    if camera_controller is None:
        print("ERROR: 'Camera Controller' object not found in Blender file.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
