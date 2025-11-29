#!/usr/bin/env python3
"""
Add SVG coordinates to noteOn events in a JSON file.

This script reads a JSON file containing note events and an SVG file containing
note head paths, then adds "svgX" and "svgY" fields to each noteOn event
representing the center of the corresponding SVG note head path.

The coordinates are calculated to match Blender's SVG import behavior:
- The path's moveto point is at the left-center of the note head ellipse
- We add half the ellipse width (~17.35 units) to get the X center
- The Y coordinate is already at the vertical center

Usage: ./add_svg_coords.py <json_file> <svg_file>
"""

import sys
import json
import re
from xml.etree import ElementTree as ET
from typing import List, Tuple, Dict, Any


# Note head ellipse dimensions (approximate, from the SVG paths)
# The path starts at the left edge of the ellipse
NOTE_HEAD_HALF_WIDTH = 17.35

# Note names in standard music notation
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_note_to_name(midi_note: int) -> str:
    """
    Convert a MIDI note number to standard music notation.
    MIDI note 60 = C4 (middle C), note 69 = A4 (440 Hz)
    """
    octave = (midi_note // 12) - 1
    note_index = midi_note % 12
    return f"{NOTE_NAMES[note_index]}{octave}"


def compute_path_center(d: str) -> Tuple[float, float]:
    """
    Compute the center of an SVG note head path.

    The note head paths use a moveto command at the left-center of the ellipse,
    followed by bezier curves that draw the ellipse. We extract the moveto
    coordinates and add half the ellipse width to get the true center X.

    Returns (center_x, center_y) or (None, None) if parsing fails.
    """
    # Parse the moveto command: "m X,Y ..." or "M X,Y ..."
    match = re.match(r'm\s+([-\d.]+),([-\d.]+)', d, re.IGNORECASE)
    if not match:
        return (None, None)

    start_x = float(match.group(1))
    start_y = float(match.group(2))

    # The path starts at the left edge of the ellipse
    # Add half-width to get the X center
    # The Y is already at the vertical center
    center_x = start_x + NOTE_HEAD_HALF_WIDTH
    center_y = start_y

    return (center_x, center_y)


def parse_svg_paths(svg_file: str) -> List[Tuple[float, float]]:
    """
    Parse all path elements from an SVG file and return their centers.
    Returns a list of (center_x, center_y) tuples, sorted by x position.

    These coordinates match what Blender imports when loading the SVG.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Handle namespace
    namespace = ''
    if root.tag.startswith('{'):
        namespace = root.tag.split('}')[0] + '}'

    centers = []
    for elem in root.iter():
        tag_name = elem.tag.replace(namespace, '')
        if tag_name == 'path':
            d_attr = elem.get('d', '')
            if d_attr:
                center_x, center_y = compute_path_center(d_attr)
                if center_x is not None:
                    centers.append((center_x, center_y))

    # Sort by X position (left to right = time order)
    centers.sort(key=lambda c: c[0])
    return centers


def collect_note_on_events(data: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Collect all noteOn events from the JSON data structure.
    Returns a list of references to noteOn event dictionaries.
    """
    note_on_events = []
    for track in data:
        for event in track:
            if event.get('type') == 'noteOn':
                note_on_events.append(event)
    return note_on_events


def add_note_names(data: List[List[Dict[str, Any]]]) -> None:
    """
    Add 'name' field with standard music notation to all noteOn and noteOff events.
    Modifies the data in place.
    """
    for track in data:
        for event in track:
            if event.get('type') in ('noteOn', 'noteOff'):
                midi_note = event.get('note')
                if midi_note is not None:
                    event['name'] = midi_note_to_name(midi_note)


def add_svg_coordinates(json_file: str, svg_file: str) -> str:
    """
    Add svgX and svgY coordinates to all noteOn events.
    Returns the modified JSON as a string.
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Add note names to all noteOn and noteOff events
    add_note_names(data)

    # Collect all noteOn events
    note_on_events = collect_note_on_events(data)

    if not note_on_events:
        return json.dumps(data, indent=2)

    # Sort noteOn events by time, then by note (for events at the same time)
    note_on_events_sorted = sorted(note_on_events, key=lambda e: (e.get('time', 0), e.get('note', 0)))

    # Get time and note ranges from noteOn events
    times = [e.get('time', 0) for e in note_on_events_sorted]
    notes = [e.get('note', 0) for e in note_on_events_sorted]

    min_time, max_time = min(times), max(times)
    min_note, max_note = min(notes), max(notes)

    # Parse SVG paths
    svg_centers = parse_svg_paths(svg_file)

    if not svg_centers:
        return json.dumps(data, indent=2)

    # Get X and Y ranges from SVG paths
    svg_xs = [c[0] for c in svg_centers]
    svg_ys = [c[1] for c in svg_centers]

    min_svg_x, max_svg_x = min(svg_xs), max(svg_xs)
    min_svg_y, max_svg_y = min(svg_ys), max(svg_ys)

    # Check if we have the same number of noteOn events and SVG paths
    if len(note_on_events_sorted) != len(svg_centers):
        # Use linear interpolation based on time and note
        # Map time to X and note to Y

        time_range = max_time - min_time if max_time != min_time else 1
        note_range = max_note - min_note if max_note != min_note else 1
        svg_x_range = max_svg_x - min_svg_x if max_svg_x != min_svg_x else 1
        svg_y_range = max_svg_y - min_svg_y if max_svg_y != min_svg_y else 1

        for event in note_on_events:
            event_time = event.get('time', 0)
            event_note = event.get('note', 0)

            # Linear interpolation for X based on time
            if time_range > 0:
                t = (event_time - min_time) / time_range
                svg_x = min_svg_x + t * svg_x_range
            else:
                svg_x = min_svg_x

            # Linear interpolation for Y based on note
            # Note: In SVG, Y increases downward, but higher notes should be higher (lower Y)
            # So we invert the mapping
            if note_range > 0:
                n = (event_note - min_note) / note_range
                svg_y = max_svg_y - n * svg_y_range
            else:
                svg_y = min_svg_y

            event['svgX'] = round(svg_x, 2)
            event['svgY'] = round(svg_y, 2)
    else:
        # One-to-one mapping: sorted noteOn events correspond to sorted SVG paths
        for event, center in zip(note_on_events_sorted, svg_centers):
            event['svgX'] = round(center[0], 2)
            event['svgY'] = round(center[1], 2)

    return json.dumps(data, indent=2)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <json_file> <svg_file>", file=sys.stderr)
        sys.exit(1)

    json_file = sys.argv[1]
    svg_file = sys.argv[2]

    try:
        result = add_svg_coordinates(json_file, svg_file)
        print(result)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing SVG: {e}", file=sys.stderr)
        sys.exit(1)
    except BrokenPipeError:
        pass


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main()
