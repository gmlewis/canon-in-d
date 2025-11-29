#!/usr/bin/env python3
"""
Blender 5.0 script to create bezier curves from pre-generated JSON data.
Run this script from within Blender's scripting environment.

This script reads the JSON output from gen-note-jumping-curves-to-json.py
and creates all the bezier curves in Blender.

All configuration is read from the JSON file, so you can edit the JSON
to customize the curves without modifying this script.
"""

import bpy
import json
import sys
import os

# ============================================================================
# Configuration
# ============================================================================
# Path to the JSON file (relative to the .blend file or absolute)
JSON_FILE = "note-jumping-curves.json"


def load_curve_data(filepath):
    """Load the curve data JSON file."""
    # Try relative to blend file first, then current directory
    paths_to_try = [filepath]

    blend_dir = bpy.path.abspath("//")
    if blend_dir:
        paths_to_try.insert(0, os.path.join(blend_dir, filepath))

    for path in paths_to_try:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"Loaded curve data from: {path}")
            return data
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON file '{path}': {e}", file=sys.stderr)
            return None

    print(f"ERROR: Could not find JSON file '{filepath}'", file=sys.stderr)
    print(f"  Tried paths: {paths_to_try}", file=sys.stderr)
    return None


def setup_collection(collection_name):
    """
    Set up the collection for curves.
    - If it exists, delete all objects within it.
    - If it doesn't exist, create it.
    """
    scene = bpy.context.scene
    collection = bpy.data.collections.get(collection_name)

    if collection is not None:
        print(f"Collection '{collection_name}' exists. Removing all objects...")
        objects_to_delete = list(collection.objects)
        for obj in objects_to_delete:
            bpy.data.objects.remove(obj, do_unlink=True)
        print(f"  Removed {len(objects_to_delete)} objects.")
    else:
        print(f"Creating new collection '{collection_name}'...")
        collection = bpy.data.collections.new(collection_name)
        scene.collection.children.link(collection)

    return collection


def create_parent_empty(collection, parent_name):
    """Create an empty object to serve as parent for all curves."""
    empty = bpy.data.objects.new(parent_name, None)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.empty_display_size = 1.0
    collection.objects.link(empty)
    print(f"Created parent empty '{parent_name}'")
    return empty


def create_bezier_curve(curve_name, curve_data, config, collection, parent_empty):
    """
    Create a single bezier curve from the curve data.

    Args:
        curve_name: Name of the curve (e.g., 'curve1')
        curve_data: Dict with 'points', 'landingCount', 'bezierPointCount'
        config: Configuration dict from JSON
        collection: Blender collection to add objects to
        parent_empty: Parent empty object

    Returns:
        Dict with created objects
    """
    points = curve_data['points']

    if len(points) < 2:
        print(f"  Skipping {curve_name}: not enough points")
        return None

    # Create empty parent for this curve
    curve_parent = bpy.data.objects.new(curve_name, None)
    curve_parent.empty_display_type = 'PLAIN_AXES'
    curve_parent.empty_display_size = 0.5
    collection.objects.link(curve_parent)
    curve_parent.parent = parent_empty

    # Create the bezier curve data
    curve_data_blender = bpy.data.curves.new(name=f"{curve_name}_bezier", type='CURVE')
    curve_data_blender.dimensions = '3D'
    curve_data_blender.resolution_u = config.get('curveResolution', 12)

    # Create a single spline for the entire curve
    spline = curve_data_blender.splines.new('BEZIER')

    # Add bezier points (one already exists by default)
    num_points = len(points)
    spline.bezier_points.add(num_points - 1)

    # Get handle type from config
    handle_type = config.get('handleType', 'AUTO')

    # Set all bezier points
    for i, point in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = (point['x'], point['y'], point['z'])
        bp.handle_left_type = handle_type
        bp.handle_right_type = handle_type

    # Create the curve object
    curve_obj = bpy.data.objects.new(f"{curve_name}_curve", curve_data_blender)
    collection.objects.link(curve_obj)
    curve_obj.parent = curve_parent

    return {
        'parent': curve_parent,
        'curve': curve_obj,
        'data': curve_data_blender
    }


def main():
    print("=" * 70)
    print("Create Curves from JSON (Blender Script)")
    print("=" * 70)

    # Step 1: Load JSON data
    print(f"\nLoading '{JSON_FILE}'...")
    data = load_curve_data(JSON_FILE)

    if data is None:
        print("Aborting due to JSON load failure.")
        return

    # Extract configuration and metadata
    config = data.get('config', {})
    metadata = data.get('metadata', {})
    curves_data = data.get('curves', {})

    print(f"\nMetadata from JSON:")
    print(f"  Source file: {metadata.get('sourceFile', 'unknown')}")
    print(f"  Max concurrent notes: {metadata.get('maxConcurrentNotes', 'unknown')}")
    print(f"  Song duration: {metadata.get('songStartTime', 0):.2f}s to {metadata.get('songEndTime', 0):.2f}s")
    print(f"  Total landing points: {metadata.get('totalLandingPoints', 'unknown')}")
    print(f"  Total bezier points: {metadata.get('totalBezierPoints', 'unknown')}")

    print(f"\nConfiguration:")
    print(f"  Collection name: {config.get('collectionName', 'unknown')}")
    print(f"  Parent name: {config.get('parentName', 'unknown')}")
    print(f"  Z offset: {config.get('maxJumpingCurveZOffset', 'unknown')}")
    print(f"  Curve resolution: {config.get('curveResolution', 'unknown')}")
    print(f"  Handle type: {config.get('handleType', 'unknown')}")

    # Step 2: Set up collection
    collection_name = config.get('collectionName', 'Note Jumping Curves')
    print(f"\nSetting up collection '{collection_name}'...")
    collection = setup_collection(collection_name)

    # Step 3: Create parent empty
    parent_name = config.get('parentName', 'Note Jumping Curves Parent')
    print(f"\nCreating parent empty...")
    parent_empty = create_parent_empty(collection, parent_name)

    # Step 4: Create all bezier curves
    print(f"\nCreating {len(curves_data)} bezier curves...")
    curve_objects = {}

    for curve_name in sorted(curves_data.keys(), key=lambda x: int(x.replace('curve', ''))):
        curve_info = curves_data[curve_name]
        result = create_bezier_curve(curve_name, curve_info, config, collection, parent_empty)
        if result:
            curve_objects[curve_name] = result
            landing_count = curve_info.get('landingCount', 0)
            bezier_count = curve_info.get('bezierPointCount', 0)
            print(f"  Created {curve_name}: {landing_count} landings, {bezier_count} bezier points")

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("Curve creation complete!")
    print(f"  Collection: '{collection_name}'")
    print(f"  Parent: '{parent_name}'")
    print(f"  Curves created: {len(curve_objects)}")
    print("\nHierarchy:")
    print(f"  {collection_name} (Collection)")
    print(f"  └── {parent_name} (Empty)")
    for curve_name in sorted(curve_objects.keys(), key=lambda x: int(x.replace('curve', ''))):
        print(f"      ├── {curve_name} (Empty)")
        print(f"      │   └── {curve_name}_curve (Bezier Curve)")
    print("=" * 70)

    # Select the parent to make it easy to find
    bpy.ops.object.select_all(action='DESELECT')
    parent_empty.select_set(True)
    bpy.context.view_layer.objects.active = parent_empty
    print(f"\nSelected '{parent_name}' - use Outliner to explore the hierarchy")


if __name__ == "__main__":
    main()
