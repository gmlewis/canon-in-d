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

3. Unit tests should be run regularly by running `./test-all.sh`.

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

