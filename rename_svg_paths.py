#!/usr/bin/env python3
"""
Standalone script to rename SVG path IDs to human-readable, debuggable names.

This script reads the NoteHeads SVG file and the CanonInD.json file, matches
SVG paths to note events using the same sorted index matching as
gen-note-jumping-curves-to-json.py, and renames each path ID to include:
- The X position (zero-padded for lexicographic sorting)
- The note name (e.g., D3, A3, F#4)
- The timestamp
- A sequential index for uniqueness

Matching algorithm (same as gen-note-jumping-curves-to-json.py):
- SVG paths are sorted by (X, Y) - left to right, top to bottom within chords
- Notes are sorted by (time, -note) - chronological, higher pitch first within chords
- Matched 1-to-1 by index

Since there are 10 more SVG paths than notes, the last 10 paths will be labeled
as "extra" (these are visual elements without corresponding noteOn events).

SVG ID naming rules:
- Must start with a letter (A-Z, a-z) or underscore (_)
- Can contain letters, digits, hyphens (-), underscores (_), and periods (.)
- Cannot contain spaces or special characters like #

Example output IDs:
- note_x000497_D3_t0p00_001
- note_x001028_Fs4_t0p90_004  (F# becomes Fs since # is not valid in IDs)
- note_x143000_extra_001 (extra SVG path without matching note)
"""

import json
import re
import sys
from xml.etree import ElementTree as ET

# ============================================================================
# Configuration
# ============================================================================
INPUT_SVG_FILE = "Canon_in_D-single-svg-printing_NoteHeads.svg"
INPUT_JSON_FILE = "CanonInD.json"
OUTPUT_SVG_FILE = "Canon_in_D-single-svg-printing_NoteHeads_renamed.svg"

# Note head half-width for center calculation (same as in gen-note-jumping-curves-to-json.py)
NOTE_HEAD_HALF_WIDTH = 17.35


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


def collect_note_on_events(data):
    """
    Collect all noteOn events from the JSON data.
    Returns a list of dicts with note info, sorted by (time, -note).
    
    The sorting matches SVG path sorting by (X, Y):
    - time corresponds to X position (left to right)
    - higher note (pitch) corresponds to lower Y (top of staff)
    """
    note_on_events = []

    for track in data:
        if not isinstance(track, list):
            continue
        for event in track:
            if isinstance(event, dict) and event.get('type') == 'noteOn':
                note_on_events.append({
                    'name': event.get('name', 'Unknown'),
                    'note': event.get('note', 0),
                    'time': event.get('time', 0),
                })

    # Sort by time, then by DESCENDING note (higher pitch first)
    # This matches SVG sorting by (X, Y) where lower Y = higher pitch
    note_on_events.sort(key=lambda e: (e['time'], -e['note']))

    return note_on_events


def extract_svg_path_centers(svg_file):
    """
    Extract path elements from SVG file with their center coordinates.
    Returns a list of (element, center_x, center_y) tuples sorted by (X, Y),
    and the ElementTree for later modification.
    """
    try:
        # Register the SVG namespace to preserve it during output
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        tree = ET.parse(svg_file)
        root = tree.getroot()
    except Exception as e:
        print(f"ERROR: Failed to parse SVG file '{svg_file}': {e}", file=sys.stderr)
        return None, None

    paths = []

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

            # Calculate center (same logic as in gen-note-jumping-curves-to-json.py)
            center_x = start_x + NOTE_HEAD_HALF_WIDTH
            center_y = start_y

            paths.append((elem, center_x, center_y))

    # Sort by X position, then by Y position
    # For chords (same X), lower Y = higher pitch, matching notes sorted by -note
    paths.sort(key=lambda p: (p[1], p[2]))

    return paths, tree


def sanitize_note_name(name):
    """
    Convert note name to a valid SVG ID component.
    - Replace # with 's' (sharp)
    - Replace b with 'b' (flat) - already valid
    - Remove any other invalid characters
    """
    # Replace sharp symbol
    name = name.replace('#', 's')
    # Remove any characters that aren't alphanumeric
    name = re.sub(r'[^A-Za-z0-9]', '', name)
    return name


def generate_path_id(note_name, center_x, timestamp, index, x_padding=6):
    """
    Generate a valid, human-readable SVG path ID.

    Format: note_x<X>_<NoteName>_t<Time>_<Index>

    X is placed first (after note_) and zero-padded for lexicographic sorting.

    Example: note_x001028_Fs4_t0p90_004
    """
    safe_name = sanitize_note_name(note_name)
    x_int = int(round(center_x))
    x_str = f"{x_int:0{x_padding}d}"  # Zero-pad X value
    time_str = f"{timestamp:.2f}".replace('.', 'p')  # Replace . with p for validity
    index_str = f"{index:04d}"  # 4 digits for up to 9999 notes

    return f"note_x{x_str}_{safe_name}_t{time_str}_{index_str}"


def main():
    print("=" * 70)
    print("SVG Path Renamer")
    print("=" * 70)

    # Step 1: Load JSON data
    print(f"\nLoading '{INPUT_JSON_FILE}'...")
    json_data = load_json_data(INPUT_JSON_FILE)
    if json_data is None:
        return 1

    # Step 2: Collect noteOn events (sorted by time, -note)
    print("Collecting noteOn events...")
    note_events = collect_note_on_events(json_data)
    print(f"  Found {len(note_events)} noteOn events")

    # Step 3: Parse SVG and extract paths (sorted by X, Y)
    print(f"\nParsing '{INPUT_SVG_FILE}'...")
    paths_data, tree = extract_svg_path_centers(INPUT_SVG_FILE)
    if paths_data is None:
        return 1
    print(f"  Found {len(paths_data)} path elements")

    # Step 4: Match paths to notes by sorted index
    print("\nMatching paths to notes by sorted index...")
    
    if len(paths_data) != len(note_events):
        diff = len(paths_data) - len(note_events)
        print(f"  Note: {abs(diff)} {'extra SVG paths' if diff > 0 else 'missing SVG paths'}")
        print(f"    SVG paths: {len(paths_data)}")
        print(f"    Note events: {len(note_events)}")

    # Calculate the X padding needed to cover the full range
    max_x = max(center_x for elem, center_x, center_y in paths_data)
    x_padding = len(str(int(round(max_x))))
    print(f"  Max X value: {int(round(max_x))} (using {x_padding} digit padding)")

    # Step 5: Rename all paths
    print("\nRenaming paths...")
    
    renamed_count = 0
    matched_count = 0
    extra_count = 0
    
    for i, (elem, center_x, center_y) in enumerate(paths_data):
        old_id = elem.get('id', 'no-id')

        if i < len(note_events):
            # Match by sorted index
            note = note_events[i]
            new_id = generate_path_id(
                note['name'],
                center_x,
                note['time'],
                i + 1,
                x_padding
            )
            matched_count += 1
        else:
            # Extra SVG path without matching note
            extra_count += 1
            x_str = f"{int(round(center_x)):0{x_padding}d}"
            new_id = f"note_x{x_str}_extra_{extra_count:03d}"

        elem.set('id', new_id)
        renamed_count += 1

        # Print progress every 100 paths
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1:4d}] {old_id} -> {new_id}")

    print(f"\n  Renamed {renamed_count} paths")
    print(f"  Matched: {matched_count}")
    print(f"  Extra: {extra_count}")

    # Step 6: Write output SVG
    print(f"\nWriting '{OUTPUT_SVG_FILE}'...")
    try:
        tree.write(OUTPUT_SVG_FILE, encoding='unicode', xml_declaration=False)

        import os
        file_size = os.path.getsize(OUTPUT_SVG_FILE)
        print(f"  Successfully wrote {file_size:,} bytes")

    except Exception as e:
        print(f"ERROR: Failed to write output file: {e}", file=sys.stderr)
        return 1

    # Step 7: Summary
    print("\n" + "=" * 70)
    print("Renaming complete!")
    print(f"  Input SVG:  {INPUT_SVG_FILE}")
    print(f"  Output SVG: {OUTPUT_SVG_FILE}")
    print(f"  Paths renamed: {renamed_count}")
    print(f"  Notes matched: {matched_count}")
    print(f"  Extra paths: {extra_count}")
    print(f"\nID Format: note_x<X>_<NoteName>_t<Time>_<Index>")
    print(f"  X is zero-padded to {x_padding} digits for lexicographic sorting")
    print(f"  Example: note_x001028_Fs4_t0p90_004")
    print("  (# replaced with 's', decimal point replaced with 'p')")
    if extra_count > 0:
        print(f"  Extra paths (no matching note): note_x<X>_extra_<N>")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
