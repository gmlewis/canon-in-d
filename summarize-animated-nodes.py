import bpy

def report_animated_properties():
    """
    Traverse all objects in the current scene and report all animated properties
    with their full property names and keyframe counts.

    This script works with Blender 5.0's new layered action system.
    """

    print("\n" + "="*80)
    print("ANIMATED PROPERTIES REPORT - Blender 5.0")
    print("="*80 + "\n")

    animated_count = 0
    total_keyframes = 0

    # Iterate through all objects in the current scene
    for obj in bpy.context.scene.objects:
        # Check if object has animation data
        if obj.animation_data is None:
            continue

        # Get the action
        action = obj.animation_data.action
        if action is None:
            continue

        # Get the assigned slot
        slot = obj.animation_data.action_slot
        if slot is None:
            continue

        print(f"Object: {obj.name} (Type: {obj.type})")
        print(f"  Action: {action.name}")
        print(f"  Slot: {slot.name_display}")
        print("-" * 70)

        # In Blender 5.0, actions have layers, which contain strips, which contain channelbags
        # Currently limited to one layer with one strip
        if not action.layers:
            print("  No animation layers found.\n")
            continue

        for layer_idx, layer in enumerate(action.layers):
            for strip_idx, strip in enumerate(layer.strips):
                # Get the channelbag for this slot
                try:
                    channelbag = strip.channelbag(slot)
                except:
                    # Channelbag doesn't exist for this slot
                    continue

                if channelbag is None:
                    continue

                # Iterate through all F-Curves in the channelbag
                for fcurve in channelbag.fcurves:
                    data_path = fcurve.data_path
                    array_index = fcurve.array_index
                    num_keyframes = len(fcurve.keyframe_points)

                    # Build full property name
                    if array_index >= 0:
                        # Property with array index (e.g., location[0])
                        full_property = f"{data_path}[{array_index}]"
                    else:
                        # Single property (e.g., hide_viewport)
                        full_property = data_path

                    print(f"  Property: {full_property}")
                    print(f"    Keyframes: {num_keyframes}")

                    # Optional: Show keyframe frames
                    if num_keyframes > 0 and num_keyframes <= 10:
                        frames = [kf.co[0] for kf in fcurve.keyframe_points]
                        print(f"    Frames: {frames}")

                    animated_count += 1
                    total_keyframes += num_keyframes

        print()  # Blank line between objects

    # Check other data types that can be animated
    # Materials
    for mat in bpy.data.materials:
        if mat.animation_data is None or mat.animation_data.action is None:
            continue

        action = mat.animation_data.action
        slot = mat.animation_data.action_slot

        if slot is None:
            continue

        print(f"Material: {mat.name}")
        print(f"  Action: {action.name}")
        print(f"  Slot: {slot.name_display}")
        print("-" * 70)

        if action.layers:
            for layer in action.layers:
                for strip in layer.strips:
                    try:
                        channelbag = strip.channelbag(slot)
                    except:
                        continue

                    if channelbag is None:
                        continue

                    for fcurve in channelbag.fcurves:
                        data_path = fcurve.data_path
                        array_index = fcurve.array_index
                        num_keyframes = len(fcurve.keyframe_points)

                        if array_index >= 0:
                            full_property = f"{data_path}[{array_index}]"
                        else:
                            full_property = data_path

                        print(f"  Property: {full_property}")
                        print(f"    Keyframes: {num_keyframes}")

                        animated_count += 1
                        total_keyframes += num_keyframes
            print()

    # Cameras
    for cam in bpy.data.cameras:
        if cam.animation_data is None or cam.animation_data.action is None:
            continue

        action = cam.animation_data.action
        slot = cam.animation_data.action_slot

        if slot is None:
            continue

        print(f"Camera: {cam.name}")
        print(f"  Action: {action.name}")
        print(f"  Slot: {slot.name_display}")
        print("-" * 70)

        if action.layers:
            for layer in action.layers:
                for strip in layer.strips:
                    try:
                        channelbag = strip.channelbag(slot)
                    except:
                        continue

                    if channelbag is None:
                        continue

                    for fcurve in channelbag.fcurves:
                        data_path = fcurve.data_path
                        array_index = fcurve.array_index
                        num_keyframes = len(fcurve.keyframe_points)

                        if array_index >= 0:
                            full_property = f"{data_path}[{array_index}]"
                        else:
                            full_property = data_path

                        print(f"  Property: {full_property}")
                        print(f"    Keyframes: {num_keyframes}")

                        animated_count += 1
                        total_keyframes += num_keyframes
            print()

    # Lights
    for light in bpy.data.lights:
        if light.animation_data is None or light.animation_data.action is None:
            continue

        action = light.animation_data.action
        slot = light.animation_data.action_slot

        if slot is None:
            continue

        print(f"Light: {light.name}")
        print(f"  Action: {action.name}")
        print(f"  Slot: {slot.name_display}")
        print("-" * 70)

        if action.layers:
            for layer in action.layers:
                for strip in layer.strips:
                    try:
                        channelbag = strip.channelbag(slot)
                    except:
                        continue

                    if channelbag is None:
                        continue

                    for fcurve in channelbag.fcurves:
                        data_path = fcurve.data_path
                        array_index = fcurve.array_index
                        num_keyframes = len(fcurve.keyframe_points)

                        if array_index >= 0:
                            full_property = f"{data_path}[{array_index}]"
                        else:
                            full_property = data_path

                        print(f"  Property: {full_property}")
                        print(f"    Keyframes: {num_keyframes}")

                        animated_count += 1
                        total_keyframes += num_keyframes
            print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total animated properties: {animated_count}")
    print(f"Total keyframes: {total_keyframes}")
    print()

# Run the report
if __name__ == "__main__":
    report_animated_properties()
