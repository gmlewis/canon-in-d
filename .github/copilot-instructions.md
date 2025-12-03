# Repository Instructions for GitHub Copilot

## 🚨 CRITICAL: DO NOT USE `run_in_terminal` 🚨

The `run_in_terminal` tool is **BROKEN** in this environment and causes the agent to hang indefinitely.

**YOU MUST ALWAYS USE `create_and_run_task` INSTEAD.**

### Rules for Command Execution:
1.  **NEVER** use `run_in_terminal`.
2.  **ALWAYS** use `create_and_run_task` for ALL shell commands, including:
    *   Running tests (e.g., `./test-all.sh`, `pnpm test`)
    *   Listing files (e.g., `ls -F`)
    *   Checking file contents (if `read_file` is not suitable)
    *   Building the project
    *   Any other shell interaction.

### Preferred Tools:
*   Use `list_dir` to list files in a directory.
*   Use `read_file` to read file contents.
*   Use `create_and_run_task` for everything else that requires execution.

# Copilot LLM Agent Rules and information for this repo

1. **CRITICAL**: Never use any `git` commands except for the following:
   - git diff ...
   - git show ...
   - git log ...

2. This repo represents a Blender project. All your work will be done
   in two kinds of Python3 files:
   - Blender scripts that use the `bpy` module to manipulate Blender objects.
   - Regular Python scripts that do not use the `bpy` module and are
     invokable from the command line with bash "shebang" (e.g. `#!/usr/bin/env python3`).

3. Unit tests should be run regularly by running `./test-all.sh`
   (this script invokes `pytest curve_generator_tests.py`).

4. Please remember that when "bouncing arcs" are discussed, the bounces
   always happen in the Blender Z axis (up/down), never in X or Y. This
   means that from an orhtographic top view, all arcs appear as straight
   lines and their midpoints are always the average of the start and end
   points of the arc in X and Y. The Z cooridinate of the start, mid, and
   end points are always hard-coded and should not be modified.

5. Part of the design requirement for these "bouncing arcs" is that they
   have an initial hard-coded "fly-in" from off-screen on the left to the
   first landing note position. Additionally, they have a hard-coded
   "fly-out" from the last landing note position to off-screen on the right.
   Do not modify these behaviors, and for the curve algorithms concentrate
   on the landing points and their midpoints only and ignore the fly-in and\
   fly-out segments.

6. Current regression focus: behavior is solid through note
   `note_x105151_B3_t180p90_1217`, but after ~180.9s a number of tests
   intentionally fail (missing landings and ordering issues).
   Preserve the pre-180s success while iterating on fixes for the later
   section, and keep those diagnostic tests intact.

7. The final two chords in the sheet music (notes D2, A2, F#3, D4, F#4, A4, D5)
   are tied together. There is no second set of `noteOn` events for this tied
   repeat before the music moves back to the preceding C#4, A4, C#5 chords.
   Keep this in mind when building or debugging the curve generator: the SVG
   still has two identical clusters of note heads, but the MIDI data only has
   one onset, so any algorithm must account for those missing `noteOn` events
   to land every note in that section.

8. Never modify these source-of-truth data files:
   - CanonInD.json
   - Canon_in_D-single-svg-printing_NoteHeads.svg
