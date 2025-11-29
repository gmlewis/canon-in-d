#!/usr/bin/env python3
import sys
import zipfile
import xml.etree.ElementTree as ET
import json
import math
import os

# Note names for MIDI note number conversion
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_note_to_name(note_number):
    """
    Convert a MIDI note number to a note name.

    MIDI note 60 = C4 (middle C)
    MIDI note 69 = A4 (concert pitch 440 Hz)

    Returns a string like 'C4', 'F#5', 'Bb3', etc.
    """
    octave = (note_number // 12) - 1
    note_index = note_number % 12
    return f"{NOTE_NAMES[note_index]}{octave}"

def parse_dawproject(file_path):
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open('project.xml') as f:
            root = ET.fromstring(f.read())
    return root

def get_tempo_map(root):
    transport = root.find('Transport')
    initial_bpm = 120.0
    tempo_id = None
    if transport is not None:
        tempo_elem = transport.find('Tempo')
        if tempo_elem is not None:
            initial_bpm = float(tempo_elem.attrib.get('value', 120))
            tempo_id = tempo_elem.attrib.get('id')

    points = []
    # Find automation for tempo_id
    # Search for any element with a Target parameter=tempo_id
    for elem in root.iter():
        target = elem.find('Target')
        if target is not None and target.attrib.get('parameter') == tempo_id:
            # Found automation
            # Look for Points or RealPoint children
            # The structure observed was <TempoAutomation ...> <Target .../> <RealPoint .../> ... </TempoAutomation>
            # So RealPoints are children of elem
            for child in elem:
                if child.tag == 'RealPoint':
                    time = float(child.attrib.get('time'))
                    value = float(child.attrib.get('value'))
                    points.append((time, value))

            # Also check for Points container
            pts_container = elem.find('Points')
            if pts_container is not None:
                for pt in pts_container:
                    time = float(pt.attrib.get('time'))
                    value = float(pt.attrib.get('value'))
                    points.append((time, value))
            break # Assume only one automation curve for tempo

    points.sort(key=lambda x: x[0])
    return initial_bpm, points

def beats_to_seconds(beats, initial_bpm, tempo_points):
    # Integrate time
    # tempo_points is list of (beat, bpm)
    # We assume piecewise linear BPM (or constant if start/end match)

    current_time = 0.0
    current_beat = 0.0
    current_bpm = initial_bpm

    # Filter points that are after the requested beats?
    # No, we need to integrate up to 'beats'.

    # We iterate through segments
    # Merge points into a list of events: (beat, new_bpm)
    # Note: points might have multiple values at same beat (step change)
    # We process them in order.

    # To handle integration correctly, we need segments [b1, b2] with start_bpm and end_bpm.

    # Construct segments
    segments = []
    last_beat = 0.0
    last_bpm = initial_bpm

    for pt_beat, pt_bpm in tempo_points:
        if pt_beat > last_beat:
            segments.append({
                'start': last_beat,
                'end': pt_beat,
                'bpm_start': last_bpm,
                'bpm_end': last_bpm # Assumption: value holds until next point unless it's a ramp?
                # Wait, in the observed data:
                # 63.0 -> 47
                # 63.5625 -> 47
                # This implies the value at 63.0 SETS the bpm.
                # And the value at 63.5625 CONFIRMS it (or changes it for the next segment).
                # But if we have:
                # 63.0 -> 47
                # 64.0 -> 50
                # Is it constant 47 or ramp to 50?
                # The attribute 'interpolation'="linear" suggests ramp.
                # But in the observed data, the start and end values of segments matched.
                # So we can assume linear ramp between points.
            })
            # But wait, if we have a step change at 63.0 (two points), the segment length is 0.
            # My loop handles pt_beat > last_beat.

        # If pt_beat == last_beat, we just update the bpm (step change)
        last_beat = pt_beat
        last_bpm = pt_bpm

    # Add final segment to infinity (or enough to cover 'beats')
    segments.append({
        'start': last_beat,
        'end': float('inf'),
        'bpm_start': last_bpm,
        'bpm_end': last_bpm
    })

    # Now integrate
    total_time = 0.0
    remaining_beats = beats

    # We need to find the time for 'beats'
    # We can just sum up segments until we cover 'beats'

    for seg in segments:
        if beats <= seg['start']:
            break

        seg_end = seg['end']
        if beats < seg_end:
            duration_beats = beats - seg['start']
            # Calculate time for this partial segment
            # BPM goes from bpm_start to interpolated value
            # fraction = duration_beats / (seg['end'] - seg['start'])
            # bpm_target = bpm_start + (bpm_end - bpm_start) * fraction
            # But wait, we need the bpm at 'beats'.

            # Actually, we need to know if the segment in the XML defines the slope.
            # In the XML:
            # Point at T1, V1
            # Point at T2, V2
            # Segment T1->T2 has BPM going from V1 to V2.

            # My segment construction above:
            # last_bpm is the value of the LAST point processed.
            # So for segment between point A and point B:
            # Start BPM = value of A.
            # End BPM = value of B.

            # Let's refine segment construction.
            pass

    # Re-construct segments properly
    # List of (beat, bpm)
    # If multiple points at same beat, the last one determines the start BPM for the next interval.
    # The first one determines the end BPM of the previous interval?
    # No, usually automation is:
    # Point A (t1, v1)
    # Point B (t2, v2)
    # Segment t1-t2 goes from v1 to v2.

    # If we have step:
    # Point A (t1, v1)
    # Point B (t1, v2)
    # Segment length 0. Instant change.

    # So we just need the list of points.
    # Add initial point (0, initial_bpm) if not present.

    sorted_points = []
    if not tempo_points or tempo_points[0][0] > 0:
        sorted_points.append((0.0, initial_bpm))
    sorted_points.extend(tempo_points)

    total_time = 0.0

    for i in range(len(sorted_points) - 1):
        t1, v1 = sorted_points[i]
        t2, v2 = sorted_points[i+1]

        if beats < t1:
            break # Should not happen if we start at 0

        if beats >= t2:
            # Full segment
            dur = t2 - t1
            if dur > 0:
                if abs(v2 - v1) < 0.0001:
                    total_time += dur * 60.0 / v1
                else:
                    # Integral of 60 / (v1 + slope*t) dt
                    # = (60/slope) * ln(v2/v1)
                    slope = (v2 - v1) / dur
                    total_time += (60.0 / slope) * math.log(v2 / v1)
        else:
            # Partial segment
            dur = beats - t1
            if dur > 0:
                # Calculate v_end at 'beats'
                slope = (v2 - v1) / (t2 - t1)
                v_end = v1 + slope * dur

                if abs(slope) < 0.0001:
                    total_time += dur * 60.0 / v1
                else:
                    total_time += (60.0 / slope) * math.log(v_end / v1)
            return total_time

    # If beats is beyond last point
    last_t, last_v = sorted_points[-1]
    if beats > last_t:
        dur = beats - last_t
        total_time += dur * 60.0 / last_v

    return total_time

def main():
    if len(sys.argv) < 2:
        print("Usage: ./dawproject_to_json.py <input_dawproject_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}.json"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    root = parse_dawproject(input_file)
    initial_bpm, tempo_points = get_tempo_map(root)

    # Parse Tracks
    structure = root.find('Structure')
    tracks = {} # id -> name
    if structure is not None:
        for track in structure.findall('Track'):
            tracks[track.attrib['id']] = track.attrib.get('name')

    # Parse Notes
    arrangement = root.find('Arrangement')
    track_events = {} # track_name -> list of events

    # Transposition map (semitones)
    # Contrabass sounds an octave lower than written (-12 semitones)
    transposition_map = {
        "Contrabass": -12
    }

    if arrangement is not None:
        lanes_container = arrangement.find('Lanes')
        if lanes_container is not None:
            for lane in lanes_container.findall('Lanes'):
                track_id = lane.attrib.get('track')
                track_name = tracks.get(track_id)
                if not track_name:
                    continue

                if track_name not in track_events:
                    track_events[track_name] = []

                events = track_events[track_name]

                # Get transposition for this track
                transpose = transposition_map.get(track_name, 0)

                # Add track name meta event
                # Only add if empty (first time)
                if not events:
                    events.append({
                        "absoluteTicks": 0,
                        "channel": 15, # Arbitrary
                        "type": "meta",
                        "metaType": 3,
                        "subType": "trackName",
                        "text": track_name,
                        "time": 0.0
                    })

                for clips in lane.findall('Clips'):
                    for clip in clips.findall('Clip'):
                        clip_start = float(clip.attrib.get('time', 0))
                        clip_play_start = float(clip.attrib.get('playStart', 0))

                        notes_container = clip.find('Notes')
                        if notes_container is not None:
                            for note in notes_container.findall('Note'):
                                note_start = float(note.attrib.get('time', 0))
                                note_dur = float(note.attrib.get('duration', 0))
                                key = int(note.attrib.get('key', 60)) + transpose
                                vel = float(note.attrib.get('vel', 0.5))
                                rel = float(note.attrib.get('rel', 0.5))

                                abs_beat_start = clip_start + (note_start - clip_play_start)
                                abs_beat_end = abs_beat_start + note_dur

                                time_start = beats_to_seconds(abs_beat_start, initial_bpm, tempo_points)
                                time_end = beats_to_seconds(abs_beat_end, initial_bpm, tempo_points)

                                ticks_start = int(abs_beat_start * 960)
                                ticks_end = int(abs_beat_end * 960)

                                # Note On
                                events.append({
                                    "absoluteTicks": ticks_start,
                                    "channel": 13, # Arbitrary
                                    "type": "noteOn",
                                    "note": key,
                                    "name": midi_note_to_name(key),
                                    "velocity": int(vel * 127),
                                    "vel": vel,
                                    "time": time_start
                                })

                                # Note Off
                                events.append({
                                    "absoluteTicks": ticks_end,
                                    "channel": 13, # Arbitrary
                                    "type": "noteOff",
                                    "note": key,
                                    "name": midi_note_to_name(key),
                                    "velocity": int(rel * 127),
                                    "rel": rel,
                                    "time": time_end
                                })    # Sort events and add endOfTrack
    final_output = []

    # We want to output tracks in a specific order?
    # The JSON has Violin I, Violin II, Viola, Violoncello, Contrabass.
    # We can sort by track name or just append.

    ordered_tracks = ["Violin I", "Violin II", "Viola", "Violoncello", "Contrabass"]
    # Add others if any
    for t in track_events:
        if t not in ordered_tracks:
            ordered_tracks.append(t)

    for t_name in ordered_tracks:
        if t_name in track_events:
            evts = track_events[t_name]
            evts.sort(key=lambda x: (x['absoluteTicks'], x['type'] == 'noteOff')) # Sort noteOn before noteOff if same tick? No, usually noteOff first?
            # Actually, if noteOff and noteOn are at same time (legato), noteOff should come first usually.
            # 'noteOff' > 'noteOn' string-wise? No.
            # Let's sort by ticks. Stable sort will keep order if ticks equal.
            # But we want deterministic order.

            # Find max time for endOfTrack
            max_time = 0
            max_ticks = 0
            if evts:
                last_evt = max(evts, key=lambda x: x['time'])
                max_time = last_evt['time']
                max_ticks = last_evt['absoluteTicks']

            evts.append({
                "absoluteTicks": max_ticks,
                "channel": 15,
                "type": "meta",
                "metaType": 47,
                "subType": "endOfTrack",
                "time": max_time
            })

            final_output.append(evts)

    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"Converted {input_file} to {output_file}")

if __name__ == '__main__':
    main()
