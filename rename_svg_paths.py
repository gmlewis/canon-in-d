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
from collections import defaultdict, deque
from xml.etree import ElementTree as ET

# ============================================================================
# Configuration
# ============================================================================
INPUT_SVG_FILE = "Canon_in_D-single-svg-printing_NoteHeads.svg"
INPUT_JSON_FILE = "CanonInD.json"
OUTPUT_SVG_FILE = "Canon_in_D-single-svg-printing_NoteHeads_renamed.svg"
MIDI_CHORD_TIME_TOLERANCE = 0.03

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
    Collect all noteOn events grouped by timestamp.

    Returns a list of dicts: {
        'time': float,
        'notes': [{'name': str, 'note': int, 'time': float}, ...]
    }
    with each group's notes sorted by descending pitch so they align with
    SVG chord ordering (top-to-bottom).
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
                    'time': event.get('time', 0.0)
                })

    if not note_on_events:
        return []

    note_on_events.sort(key=lambda e: e['time'])

    midi_chords = []
    current_group = [note_on_events[0]]
    group_time = note_on_events[0]['time']

    for event in note_on_events[1:]:
        if event['time'] - current_group[-1]['time'] <= MIDI_CHORD_TIME_TOLERANCE:
            current_group.append(event)
        else:
            notes_sorted = sorted(current_group, key=lambda n: -n['note'])
            midi_chords.append({'time': group_time, 'notes': notes_sorted})
            current_group = [event]
            group_time = event['time']

    notes_sorted = sorted(current_group, key=lambda n: -n['note'])
    midi_chords.append({'time': group_time, 'notes': notes_sorted})

    return midi_chords


def group_by_x_tolerance(items, get_x, tolerance=15.0):
    """
    Group items by X position with tolerance.
    Items within 'tolerance' X units of each other are considered the same chord.

    Returns a list of groups, where each group is sorted by the average X of the group.
    Within each group, items retain their original order.
    """
    if not items:
        return []

    sorted_items = sorted(items, key=get_x)

    groups = []
    current_group = [sorted_items[0]]
    current_group_start_x = get_x(sorted_items[0])

    for item in sorted_items[1:]:
        item_x = get_x(item)
        if item_x - current_group_start_x <= tolerance:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]
            current_group_start_x = item_x

    groups.append(current_group)

    return groups


def extract_svg_path_centers(svg_file):
    """
    Extract path elements from SVG file with their center coordinates.

    Returns (paths, tree) where paths is a list of tuples
    (element, center_x, center_y) sorted left-to-right/top-to-bottom.
    """
    try:
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        tree = ET.parse(svg_file)
        root = tree.getroot()
    except Exception as e:
        print(f"ERROR: Failed to parse SVG file '{svg_file}': {e}", file=sys.stderr)
        return None, None

    paths = []

    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if tag == 'path':
            d = elem.get('d', '')
            if not d:
                continue

            match = re.match(r'm\s+([\-\d.]+),([\-\d.]+)', d, re.IGNORECASE)
            if not match:
                continue

            start_x = float(match.group(1))
            start_y = float(match.group(2))

            center_x = start_x + NOTE_HEAD_HALF_WIDTH
            center_y = start_y

            paths.append((elem, center_x, center_y))

    chord_groups = group_by_x_tolerance(paths, lambda p: p[1], tolerance=15.0)

    sorted_paths = []
    for group in chord_groups:
        group_sorted = sorted(group, key=lambda p: p[2])
        sorted_paths.extend(group_sorted)

    return sorted_paths, tree


def group_paths_into_chords(paths):
    """Group sorted path tuples into chord structures with metadata."""
    chord_groups = group_by_x_tolerance(paths, lambda p: p[1], tolerance=15.0)
    grouped = []
    for group in chord_groups:
        sorted_group = sorted(group, key=lambda p: p[2])
        avg_x = sum(p[1] for p in sorted_group) / len(sorted_group)
        grouped.append({
            'avg_x': avg_x,
            'elements': sorted_group
        })
    return grouped


def assign_ids_to_elements(elements, midi_notes, timestamp, start_index, x_padding):
    """Assign new IDs to SVG elements based on the provided MIDI notes."""
    counter = start_index
    for note, data in zip(midi_notes, elements):
        elem, center_x, _ = data
        new_id = generate_path_id(note['name'], center_x, timestamp, counter, x_padding)
        elem.set('id', new_id)
        counter += 1
    return counter


def is_tie_duplicate_group(next_group, reference_ys, reference_avg_x,
                           current_time, next_time,
                           y_tolerance=8.0, x_tolerance=600.0,
                           min_time_gap=0.75):
    """
    Determine whether the upcoming SVG chord group is a tied duplicate
    of the previously matched MIDI chord.
    """
    if next_time - current_time < min_time_gap:
        return False

    if len(next_group['elements']) != len(reference_ys):
        return False

    group_ys = [pt[2] for pt in next_group['elements']]
    avg_x = next_group['avg_x']

    if abs(avg_x - reference_avg_x) > x_tolerance:
        return False

    for ref_y, grp_y in zip(reference_ys, group_ys):
        if abs(ref_y - grp_y) > y_tolerance:
            return False

    return True


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
    midi_chords = collect_note_on_events(json_data)
    total_notes = sum(len(chord['notes']) for chord in midi_chords)
    print(f"  Found {len(midi_chords)} MIDI chord groups / {total_notes} noteOn events")

    # Step 3: Parse SVG and extract paths (sorted by X, Y)
    print(f"\nParsing '{INPUT_SVG_FILE}'...")
    paths_data, tree = extract_svg_path_centers(INPUT_SVG_FILE)
    if paths_data is None:
        return 1
    print(f"  Found {len(paths_data)} path elements")

    chord_groups = group_paths_into_chords(paths_data)
    print(f"  -> Collapsed into {len(chord_groups)} chord groups for matching")

    print("\nMatching SVG chords to MIDI chords with tie detection...")

    svg_available = sum(len(group['elements']) for group in chord_groups)
    if svg_available != total_notes:
        diff = svg_available - total_notes
        print(f"  Note: {abs(diff)} {'extra SVG note heads' if diff > 0 else 'missing SVG note heads'}")
        print(f"    SVG note heads: {svg_available}")
        print(f"    MIDI noteOns: {total_notes}")

    # Calculate the X padding needed to cover the full range
    max_x = max(center_x for elem, center_x, center_y in paths_data)
    x_padding = len(str(int(round(max_x))))
    print(f"  Max X value: {int(round(max_x))} (using {x_padding} digit padding)")

    # Step 5: Rename all paths
    print("\nRenaming paths...")

    renamed_count = 0
    matched_count = 0
    extra_count = 0
    tie_match_count = 0

    svg_chord_queue = deque(chord_groups)
    id_counter = 1

    for chord_idx, chord in enumerate(midi_chords):
        target_count = len(chord['notes'])
        combined_elements = []

        while svg_chord_queue and len(combined_elements) < target_count:
            group = svg_chord_queue.popleft()
            combined_elements.extend(group['elements'])

        if not combined_elements:
            print(f"  WARNING: Ran out of SVG note heads for MIDI chord at t={chord['time']:.3f}s")
            break

        combined_elements.sort(key=lambda item: item[2])
        assigned = combined_elements[:target_count]
        id_counter = assign_ids_to_elements(assigned, chord['notes'],
                                            chord['time'], id_counter, x_padding)
        renamed_count += len(assigned)
        matched_count += len(assigned)

        leftover = combined_elements[target_count:]
        if leftover:
            avg_x = sum(item[1] for item in leftover) / len(leftover)
            svg_chord_queue.appendleft({'avg_x': avg_x, 'elements': leftover})

        reference_ys = [item[2] for item in assigned]
        reference_avg_x = sum(item[1] for item in assigned) / len(assigned)
        next_midi_time = chord['time']
        if chord_idx + 1 < len(midi_chords):
            next_midi_time = midi_chords[chord_idx + 1]['time']

        # Detect chord-sized tied duplicates immediately following this chord
        while svg_chord_queue:
            next_group = svg_chord_queue[0]
            if not is_tie_duplicate_group(
                next_group,
                reference_ys,
                reference_avg_x,
                chord['time'],
                next_midi_time
            ):
                break

            svg_chord_queue.popleft()
            tie_elements = next_group['elements']
            id_counter = assign_ids_to_elements(tie_elements, chord['notes'],
                                                chord['time'], id_counter, x_padding)
            renamed_count += len(tie_elements)
            matched_count += len(tie_elements)
            tie_match_count += 1

    # Any remaining groups are unmatched extras
    while svg_chord_queue:
        leftover_group = svg_chord_queue.popleft()
        for elem, center_x, _ in leftover_group['elements']:
            extra_count += 1
            x_str = f"{int(round(center_x)):0{x_padding}d}"
            new_id = f"note_x{x_str}_extra_{extra_count:03d}"
            elem.set('id', new_id)
            renamed_count += 1

    print(f"\n  Renamed {renamed_count} paths")
    print(f"  Matched: {matched_count}")
    print(f"  Extra: {extra_count}")
    if tie_match_count:
        print(f"  Detected {tie_match_count} tied chord group(s)")

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
