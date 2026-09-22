"""Auxiliary SceneShift with original-source OT/Flow and source-side FM backprop."""

from __future__ import annotations

import sys

from train_houston_flow_transport import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--scene-shift",
        "--scene-shift-aux-original-flow",
        "--fm-backprop-source",
    ]
    main()
