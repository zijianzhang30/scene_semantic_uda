"""SceneShift auxiliary-view experiment with original-source OT/Flow.

This is a separate entry point for the minimal ablation requested on top of
the audited fair SceneShift implementation.  SceneShift remains an equal-
weight supervised auxiliary view, while q-only OT and class-conditional
FM-only Flow use the original source/target forward.
"""

from __future__ import annotations

import sys

from train_houston_flow_transport import main


if __name__ == "__main__":
    # Force the two properties that define this ablation.  User-supplied
    # training arguments (seed, epochs, out, variant, mode) pass through.
    sys.argv[1:1] = ["--scene-shift", "--scene-shift-aux-original-flow"]
    main()
