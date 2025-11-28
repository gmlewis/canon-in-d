#!/usr/bin/env python3
"""
Sort SVG path elements by the X position of the first coordinate in each path.
Usage: ./sort_svg_paths.py <svg_file>
"""

import sys
import re
from xml.etree import ElementTree as ET


def get_first_x_position(path_d: str) -> float:
    """Extract the X position from the first coordinate in a path's d attribute."""
    # Match the first coordinate pair after m/M command
    # Format can be: m x,y or M x,y (with optional spaces)
    match = re.match(r'[mM]\s*([-+]?\d*\.?\d+)', path_d)
    if match:
        return float(match.group(1))
    return 0.0


def sort_svg_paths(svg_file: str) -> str:
    """Read an SVG file and return it with paths sorted by X position."""
    # Parse the SVG
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Handle namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Check if the SVG uses a namespace
    if root.tag.startswith('{'):
        namespace = root.tag.split('}')[0] + '}'
    else:
        namespace = ''

    # Collect all path elements with their X positions
    paths_with_positions = []
    non_path_elements = []

    for child in list(root):
        tag_name = child.tag.replace(namespace, '')
        if tag_name == 'path':
            d_attr = child.get('d', '')
            x_pos = get_first_x_position(d_attr)
            paths_with_positions.append((x_pos, child))
        else:
            non_path_elements.append(child)

    # Sort paths by X position
    paths_with_positions.sort(key=lambda x: x[0])

    # Clear root and rebuild with sorted paths
    for child in list(root):
        root.remove(child)

    # Add non-path elements first (if any)
    for elem in non_path_elements:
        root.append(elem)

    # Add sorted paths
    for _, path in paths_with_positions:
        root.append(path)

    # Convert to string
    # Use a custom approach to preserve formatting
    output = ET.tostring(root, encoding='unicode')

    # Add XML declaration is not needed for SVG, but let's make it clean
    return output


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <svg_file>", file=sys.stderr)
        sys.exit(1)

    svg_file = sys.argv[1]

    try:
        result = sort_svg_paths(svg_file)
        print(result)
    except FileNotFoundError:
        print(f"Error: File '{svg_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing SVG: {e}", file=sys.stderr)
        sys.exit(1)
    except BrokenPipeError:
        # Handle pipe being closed (e.g., when piping to head)
        pass


if __name__ == '__main__':
    # Handle broken pipe signal gracefully
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main()
