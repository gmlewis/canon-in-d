#!/usr/bin/env python3
"""Regression tests for build-notehead-map output."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
MAP_PATH = ROOT / "notehead-map.json"

with MAP_PATH.open("r") as f:
    DATA = json.load(f)

HEADS = DATA["heads"]
HEAD_BY_ID = {head["svgId"]: head for head in HEADS}
INDEX = DATA["index"]

FINAL_EXTRA_IDS = [
    f"note_x143107_extra_{i:03d}" for i in range(1, 8)
]
FINAL_CHORD_IDS = [
    "note_x142646_D5_t241p80_1586",
    "note_x142646_A4_t241p80_1587",
    "note_x142646_Fs4_t241p80_1588",
    "note_x142646_D4_t241p80_1589",
    "note_x142646_Fs3_t241p80_1590",
    "note_x142646_A2_t241p80_1591",
    "note_x142646_D2_t241p80_1592",
]
TIE_DUPLICATE_IDS = [
    "note_x142235_Cs5_t239p70_1583",
    "note_x142235_A4_t239p70_1584",
    "note_x142235_Cs4_t239p70_1585",
]


def test_metadata_counts_match_actual_values():
    meta = DATA["metadata"]
    assert meta["totalHeads"] == len(HEADS)
    assert meta["playableHeads"] == sum(1 for h in HEADS if h["isPlayable"])
    assert meta["tieOnlyHeads"] == sum(1 for h in HEADS if h["isTieOnly"])
    assert meta["extraHeads"] == sum(1 for h in HEADS if h["isExtra"])


def test_final_duplicate_heads_marked_as_ties():
    for head_id in FINAL_EXTRA_IDS:
        head = HEAD_BY_ID[head_id]
        assert head["isTieOnly"], head_id
        assert not head["isPlayable"], head_id
        assert not head["isExtra"], head_id
        assert head["tieSourceHeadId"], head_id


def test_tie_duplicates_reference_preceding_notes():
    for head_id in TIE_DUPLICATE_IDS:
        head = HEAD_BY_ID[head_id]
        assert head["isTieOnly"], head_id
        assert head["tieSourceHeadId"], head_id
        src = HEAD_BY_ID[head["tieSourceHeadId"]]
        assert src["isPlayable"], head_id


def test_final_chord_heads_have_note_on_entries():
    for head_id in FINAL_CHORD_IDS:
        head = HEAD_BY_ID[head_id]
        assert head["isPlayable"], head_id
        note_on = head["noteOn"]
        assert note_on is not None, head_id
        assert abs(note_on["time"] - 241.798346) < 0.1 or head_id in {
            "note_x142646_A4_t241p80_1587",
            "note_x142646_D2_t241p80_1592",
        }


def test_note_id_index_points_into_heads():
    by_id = INDEX["byNoteId"]
    for head_id, idx in by_id.items():
        assert HEADS[idx]["svgId"] == head_id


def test_note_on_index_includes_final_chord_entries():
    by_note_on = INDEX["byNoteOn"]
    key = "241.798346|62"
    assert any(h == "note_x142646_D5_t241p80_1586" for h in by_note_on.get(key, []))
