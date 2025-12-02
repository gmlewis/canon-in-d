#!/usr/bin/env python3
"""
Blender 5.0 script to synx the "Camera Controller" empty node from pre-generated data.
Run this script from within Blender's scripting environment.

This script:
1. Reads the JSON output from build-notehead-map.py
   and creates animation keyframes in Blender for the "Camera Controller".
"""

import bpy
import json
import sys
import os

# ============================================================================
# Configuration
# ============================================================================
# Path to the JSON file (relative to the .blend file or absolute)
NOTEHEAD_MAP_FILE = "notehead-map.json"

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


def load_notehead_map_data(filepath):
    """Load the canonical notehead map JSON structure."""
    data = load_json_data(filepath)
    if data is None:
        return None
    if 'heads' not in data or 'metadata' not in data:
        print(f"ERROR: Notehead map '{filepath}' is missing required sections.", file=sys.stderr)
        return None
    return data


def main():
    print("=" * 70)
    print("Sync Camera Controller (Blender Script)")
    print("=" * 70)

    # Step 0: Load canonical note head mapping for scale + coordinates
    print(f"\nLoading note head map '{NOTEHEAD_MAP_FILE}'...")
    notehead_map = load_notehead_map_data(NOTEHEAD_MAP_FILE)
    if notehead_map is None:
        print("Aborting due to note head map load failure.")
        return 1


if __name__ == "__main__":
    main()
