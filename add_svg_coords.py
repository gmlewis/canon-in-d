#!/usr/bin/env python3
"""
Add SVG coordinates to noteOn events in a JSON file.

This script reads a JSON file containing note events and an SVG file containing
note head paths, then adds "svgX" and "svgY" fields to each noteOn event
representing the geometric center of the corresponding SVG path element.

Usage: ./add_svg_coords.py <json_file> <svg_file>
"""

import sys
import json
import re
from xml.etree import ElementTree as ET
from typing import List, Tuple, Dict, Any


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


def parse_path_commands(d: str) -> List[Tuple[str, List[float]]]:
    """Parse an SVG path d attribute into a list of commands with their parameters."""
    # This regex matches SVG path commands
    command_re = re.compile(r'([MmLlHhVvCcSsQqTtAaZz])\s*([^MmLlHhVvCcSsQqTtAaZz]*)')
    commands = []

    for match in command_re.finditer(d):
        cmd = match.group(1)
        params_str = match.group(2).strip()

        if params_str:
            # Split by comma or whitespace, handling negative numbers
            params = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', params_str)
            params = [float(p) for p in params]
        else:
            params = []

        commands.append((cmd, params))

    return commands


def compute_path_bounds(d: str) -> Tuple[float, float, float, float]:
    """
    Compute the bounding box of an SVG path.
    Returns (min_x, min_y, max_x, max_y).

    This is a simplified implementation that handles the common commands.
    """
    commands = parse_path_commands(d)

    if not commands:
        return (0, 0, 0, 0)

    x, y = 0.0, 0.0  # Current position
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    def update_bounds(px: float, py: float):
        nonlocal min_x, min_y, max_x, max_y
        min_x = min(min_x, px)
        min_y = min(min_y, py)
        max_x = max(max_x, px)
        max_y = max(max_y, py)

    for cmd, params in commands:
        if cmd == 'M':  # Absolute moveto
            for i in range(0, len(params), 2):
                x, y = params[i], params[i + 1]
                update_bounds(x, y)
        elif cmd == 'm':  # Relative moveto
            for i in range(0, len(params), 2):
                x += params[i]
                y += params[i + 1]
                update_bounds(x, y)
        elif cmd == 'L':  # Absolute lineto
            for i in range(0, len(params), 2):
                x, y = params[i], params[i + 1]
                update_bounds(x, y)
        elif cmd == 'l':  # Relative lineto
            for i in range(0, len(params), 2):
                x += params[i]
                y += params[i + 1]
                update_bounds(x, y)
        elif cmd == 'H':  # Absolute horizontal lineto
            for p in params:
                x = p
                update_bounds(x, y)
        elif cmd == 'h':  # Relative horizontal lineto
            for p in params:
                x += p
                update_bounds(x, y)
        elif cmd == 'V':  # Absolute vertical lineto
            for p in params:
                y = p
                update_bounds(x, y)
        elif cmd == 'v':  # Relative vertical lineto
            for p in params:
                y += p
                update_bounds(x, y)
        elif cmd == 'C':  # Absolute cubic bezier
            for i in range(0, len(params), 6):
                # Control points and end point
                update_bounds(params[i], params[i + 1])
                update_bounds(params[i + 2], params[i + 3])
                x, y = params[i + 4], params[i + 5]
                update_bounds(x, y)
        elif cmd == 'c':  # Relative cubic bezier
            for i in range(0, len(params), 6):
                update_bounds(x + params[i], y + params[i + 1])
                update_bounds(x + params[i + 2], y + params[i + 3])
                x += params[i + 4]
                y += params[i + 5]
                update_bounds(x, y)
        elif cmd in ('Z', 'z'):  # Close path
            pass

    if min_x == float('inf'):
        return (0, 0, 0, 0)

    return (min_x, min_y, max_x, max_y)


def compute_path_center(d: str) -> Tuple[float, float]:
    """Compute the geometric center of an SVG path."""
    min_x, min_y, max_x, max_y = compute_path_bounds(d)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return (center_x, center_y)


def parse_svg_paths(svg_file: str) -> List[Tuple[float, float]]:
    """
    Parse all path elements from an SVG file and return their centers.
    Returns a list of (center_x, center_y) tuples, sorted by x position.
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
                center = compute_path_center(d_attr)
                centers.append(center)

    # Sort by X position
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
