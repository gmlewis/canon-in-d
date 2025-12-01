#!/usr/bin/env python3
"""Generate a canonical mapping between CanonInD MIDI note events and SVG note heads.

The output JSON captures every SVG head (playable notes, tied duplicates, and decorative
extras) so downstream tools like gen-note-jumping-curves-to-json.py can rely on a
single source of truth without re-parsing the SVG.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

SVG_NOTE_HALF_WIDTH = 17.35
DEFAULT_SVG = "Canon_in_D-single-svg-printing_NoteHeads_renamed.svg"
DEFAULT_MIDI_JSON = "CanonInD.json"
DEFAULT_OUTPUT = "notehead-map.json"
DEFAULT_USER_SCALE = 100.0
X_GROUP_TOLERANCE = 15.0
MERGE_X_THRESHOLD = 45.0
TIE_TIME_GAP_MAX = 8.0  # seconds; generous to cover long sustains
TIME_ALIGNMENT_TOL = 0.75  # max |head_time - midi_time| to consider alignment valid

NOTE_BASE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


@dataclass
class NoteHead:
    svg_id: str
    svg_x: float
    svg_y: float
    blender_x: float
    blender_y: float
    approx_time: Optional[float]
    initial_name: Optional[str]
    midi_note: Optional[int]
    is_playable: bool = False
    is_tie_only: bool = False
    is_extra: bool = False
    tie_source_id: Optional[str] = None
    tie_source_time: Optional[float] = None
    note_on: Optional[Dict] = None
    note_off: Optional[Dict] = None

    def to_json(self) -> Dict:
        return {
            "id": self.svg_id,
            "svgId": self.svg_id,
            "svgX": self.svg_x,
            "svgY": self.svg_y,
            "blenderX": self.blender_x,
            "blenderY": self.blender_y,
            "approxTime": round_time(self.approx_time) if self.approx_time is not None else None,
            "noteName": self.note_on.get("name") if self.note_on else self.initial_name,
            "midiNote": self.midi_note,
            "isPlayable": self.is_playable,
            "isTieOnly": self.is_tie_only,
            "isExtra": self.is_extra,
            "tieSourceHeadId": self.tie_source_id,
            "tieSourceTime": round_time(self.tie_source_time) if self.tie_source_time is not None else None,
            "noteOn": self.note_on,
            "noteOff": self.note_off,
        }


def round_time(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value + 1e-12, 6)


def normalize_note_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    match = re.match(r"^([A-Ga-g])([sb#]?)(\d)$", raw)
    if not match:
        return None
    letter = match.group(1).upper()
    accidental = match.group(2)
    octave = int(match.group(3))
    if accidental.lower() == "s":
        accidental = "#"
    elif accidental == "":
        accidental = ""
    elif accidental not in {"#", "b"}:
        # Unsupported accidental
        return None
    return f"{letter}{accidental}{octave}"


def note_name_to_midi(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    match = re.match(r"^([A-G])([#b]?)(\d)$", name)
    if not match:
        return None
    letter, accidental, octave_str = match.groups()
    base = NOTE_BASE[letter]
    if accidental == "#":
        base += 1
    elif accidental == "b":
        base -= 1
    octave = int(octave_str)
    return (octave + 1) * 12 + base


def parse_time_from_id(svg_id: str) -> Optional[float]:
    match = re.search(r"_t(\d+)p(\d+)", svg_id)
    if not match:
        return None
    whole = match.group(1)
    frac = match.group(2)
    value = float(f"{whole}.{frac}")
    return value


def parse_svg_dimensions(svg_file: str) -> Tuple[float, float, float]:
    tree = ET.parse(svg_file)
    root = tree.getroot()
    viewbox = root.get("viewBox", "0 0 1000 1000").split()
    viewbox_height = float(viewbox[3])
    width_str = root.get("width", "1000mm")
    width_match = re.match(r"([\d.]+)\s*mm", width_str)
    width_mm = float(width_match.group(1)) if width_match else float(re.sub(r"[^\d.]", "", width_str))
    viewbox_width = float(viewbox[2])
    base_scale = (width_mm / viewbox_width) / 1000.0
    return base_scale, viewbox_height, viewbox_width


def svg_to_blender(svg_x: float, svg_y: float, scale: float, viewbox_height: float) -> Tuple[float, float]:
    return svg_x * scale, (viewbox_height - svg_y) * scale


def extract_note_heads(svg_file: str, scale: float, viewbox_height: float) -> List[NoteHead]:
    tree = ET.parse(svg_file)
    root = tree.getroot()
    heads: List[NoteHead] = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag != "path":
            continue
        svg_id = elem.get("id", "")
        if not svg_id.startswith("note_"):
            continue
        d_attr = elem.get("d", "")
        match = re.match(r"[mM]\s+([\-\d.]+),([\-\d.]+)", d_attr)
        if not match:
            continue
        start_x = float(match.group(1))
        start_y = float(match.group(2))
        center_x = start_x + SVG_NOTE_HALF_WIDTH
        center_y = start_y
        name_match = re.search(r"note_x\d+_([A-G][sb]?\d)", svg_id)
        name_from_id = normalize_note_name(name_match.group(1)) if name_match else None
        approx_time = parse_time_from_id(svg_id)
        blender_x, blender_y = svg_to_blender(center_x, center_y, scale, viewbox_height)
        midi_note = note_name_to_midi(name_from_id)
        heads.append(
            NoteHead(
                svg_id=svg_id,
                svg_x=center_x,
                svg_y=center_y,
                blender_x=blender_x,
                blender_y=blender_y,
                approx_time=approx_time,
                initial_name=name_from_id,
                midi_note=midi_note,
            )
        )
    heads.sort(key=lambda h: h.svg_x)
    return heads


def load_midi_data(path: str) -> List[List[Dict]]:
    with open(path, "r") as f:
        return json.load(f)


def collect_note_events(data: List[List[Dict]]) -> Tuple[List[Dict], Dict[int, float]]:
    events = []
    for track_idx, track in enumerate(data):
        for event in track:
            if event.get("type") not in {"noteOn", "noteOff"}:
                continue
            entry = dict(event)
            entry["_track"] = track_idx
            events.append(entry)
    events.sort(key=lambda e: (e.get("time", 0.0), 0 if e.get("type") == "noteOff" else 1))

    # Pair noteOn/off
    active: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    for event in events:
        key = (event.get("_track", 0), event.get("channel", -1), event.get("note", -1))
        if event.get("type") == "noteOn":
            active[key].append(event)
        else:
            if active[key]:
                start = active[key].pop(0)
                start["_noteOffTime"] = event.get("time")
                start["_noteOffTicks"] = event.get("absoluteTicks")
    note_on_events = [e for e in events if e.get("type") == "noteOn"]
    for idx, event in enumerate(note_on_events):
        event["_eventIndex"] = idx
    return note_on_events, {id(e): e.get("_noteOffTime") for e in note_on_events}


def group_by_x_tolerance(items: List[NoteHead], tolerance: float) -> List[List[NoteHead]]:
    if not items:
        return []
    sorted_items = sorted(items, key=lambda h: h.svg_x)
    groups: List[List[NoteHead]] = []
    current_group = [sorted_items[0]]
    group_start = sorted_items[0].svg_x
    for head in sorted_items[1:]:
        if head.svg_x - group_start <= tolerance:
            current_group.append(head)
        else:
            groups.append(current_group)
            current_group = [head]
            group_start = head.svg_x
    groups.append(current_group)
    return groups


def match_notes_to_heads(note_on_events: List[Dict], heads: List[NoteHead]) -> int:
    time_groups: Dict[float, List[Dict]] = defaultdict(list)
    for event in note_on_events:
        time_groups[event.get("time", 0.0)].append(event)
    midi_chords = []
    for time in sorted(time_groups.keys()):
        chord = sorted(time_groups[time], key=lambda e: -e.get("note", 0))
        midi_chords.append((time, chord))

    svg_groups = group_by_x_tolerance(heads, X_GROUP_TOLERANCE)
    svg_queue = deque()
    for group in svg_groups:
        sorted_group = sorted(group, key=lambda h: h.svg_y)
        avg_x = sum(h.svg_x for h in sorted_group) / len(sorted_group)
        svg_queue.append({"avg_x": avg_x, "heads": sorted_group})

    matched = 0
    total_mismatches = 0
    for midi_time, midi_notes in midi_chords:
        target = len(midi_notes)
        if not svg_queue:
            break
        combined: List[NoteHead] = []
        temp: List[Dict] = []
        last_avg = None
        combined_has_potential = False
        while svg_queue and len(combined) < target:
            group = svg_queue.popleft()
            temp.append(group)
            group_has_potential = any(
                head.approx_time is None or abs(head.approx_time - midi_time) <= TIME_ALIGNMENT_TOL
                for head in group["heads"]
            )
            if last_avg is not None and (group["avg_x"] - last_avg) > MERGE_X_THRESHOLD and combined and combined_has_potential:
                svg_queue.appendleft(group)
                temp.pop()
                break
            combined.extend(group["heads"])
            last_avg = group["avg_x"]
            if group_has_potential:
                combined_has_potential = True
        if not combined:
            continue
        combined.sort(key=lambda h: h.svg_y)
        usable = []
        skipped = []
        for head in combined:
            if head.approx_time is not None and midi_time is not None:
                if abs(head.approx_time - midi_time) > TIME_ALIGNMENT_TOL:
                    skipped.append(head)
                    continue
            usable.append(head)

        assign_count = min(target, len(usable))
        if assign_count != target:
            total_mismatches += 1
            if total_mismatches <= 5:
                print(
                    f"WARNING: chord mismatch at time {midi_time:.3f}s -> targets {target}, usable heads {len(usable)}"
                )
        assigned_ids = set()
        for idx in range(assign_count):
            head = usable[idx]
            event = midi_notes[idx]
            head.is_playable = True
            head.midi_note = event.get("note")
            head.initial_name = event.get("name") or head.initial_name
            on_time = round_time(event.get("time"))
            head.note_on = {
                "time": on_time,
                "absoluteTicks": event.get("absoluteTicks"),
                "velocity": event.get("velocity"),
                "channel": event.get("channel"),
                "track": event.get("_track"),
                "name": event.get("name"),
                "note": event.get("note"),
            }
            off_time = event.get("_noteOffTime")
            head.note_off = {
                "time": round_time(off_time) if off_time is not None else None,
                "absoluteTicks": event.get("_noteOffTicks"),
            }
            matched += 1
            assigned_ids.add(id(head))
        remaining = [h for h in combined if id(h) not in assigned_ids]
        leftover = remaining
        if leftover:
            avg_x = sum(h.svg_x for h in leftover) / len(leftover)
            svg_queue.appendleft({"avg_x": avg_x, "heads": leftover})
        temp.clear()
    return matched


def classify_unmatched_heads(heads: List[NoteHead]) -> None:
    playable_by_note: Dict[int, List[NoteHead]] = defaultdict(list)
    for head in sorted(heads, key=lambda h: (h.note_on.get("time") if h.note_on else float("inf"))):
        if head.is_playable and head.midi_note is not None:
            playable_by_note[head.midi_note].append(head)

    for head in heads:
        if head.is_playable:
            continue
        if "_extra_" in head.svg_id or head.initial_name is None:
            head.is_extra = True
            continue
        if head.midi_note is None:
            head.is_extra = True
            continue
        candidates = playable_by_note.get(head.midi_note, [])
        chosen = None
        for candidate in reversed(candidates):
            if head.approx_time is not None and candidate.note_on:
                delta = head.approx_time - candidate.note_on.get("time", 0.0)
                if delta < -0.001:
                    continue
                if delta > TIE_TIME_GAP_MAX:
                    continue
            chosen = candidate
            break
        if chosen:
            head.is_tie_only = True
            head.tie_source_id = chosen.svg_id
            head.tie_source_time = chosen.note_on.get("time") if chosen.note_on else None
        else:
            head.is_extra = True


def build_indexes(heads: List[NoteHead]) -> Dict[str, Dict[str, List[str]]]:
    by_note_on: Dict[str, List[str]] = defaultdict(list)
    by_note_id: Dict[str, str] = {}
    for head in heads:
        by_note_id[head.svg_id] = head.svg_id
        if head.note_on:
            key = f"{head.note_on['time']:.6f}|{head.note_on.get('note', -1)}"
            by_note_on[key].append(head.svg_id)
    return {"byNoteOn": by_note_on, "byNoteId": by_note_id}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical note-head mapping JSON.")
    parser.add_argument("--svg", default=DEFAULT_SVG, help="Renamed SVG file containing note head paths")
    parser.add_argument("--midi", default=DEFAULT_MIDI_JSON, help="CanonInD.json MIDI data")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("--user-scale", type=float, default=DEFAULT_USER_SCALE,
                        help="Additional user scale factor applied in Blender (default: 100.0)")
    args = parser.parse_args()

    base_scale, viewbox_height, _ = parse_svg_dimensions(args.svg)
    scale = base_scale * args.user_scale

    print(f"SVG scale: {scale:.10f}")
    heads = extract_note_heads(args.svg, scale, viewbox_height)
    print(f"Extracted {len(heads)} note heads from SVG")

    midi_data = load_midi_data(args.midi)
    note_on_events, _ = collect_note_events(midi_data)
    print(f"Loaded {len(note_on_events)} MIDI noteOn events")

    matched = match_notes_to_heads(note_on_events, heads)
    print(f"Matched {matched} noteOn events to SVG heads")

    classify_unmatched_heads(heads)
    playable = sum(1 for h in heads if h.is_playable)
    tie_only = sum(1 for h in heads if h.is_tie_only)
    extras = sum(1 for h in heads if h.is_extra)
    print(f"Playable heads: {playable}, tied-only: {tie_only}, extras: {extras}")

    index_data = build_indexes(heads)
    metadata = {
        "sourceMidi": os.path.basename(args.midi),
        "sourceSvg": os.path.basename(args.svg),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "svgScale": scale,
        "viewboxHeight": viewbox_height,
        "totalHeads": len(heads),
        "playableHeads": playable,
        "tieOnlyHeads": tie_only,
        "extraHeads": extras,
    }

    output = {
        "metadata": metadata,
        "heads": [head.to_json() for head in heads],
        "index": index_data,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote mapping to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
