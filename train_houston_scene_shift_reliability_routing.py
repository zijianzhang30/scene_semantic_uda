"""Reliability-routed Auxiliary SceneShift + original-source OT/Flow."""

from __future__ import annotations

import sys

from train_houston_flow_transport import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--scene-shift",
        "--scene-shift-aux-original-flow",
        "--fm-backprop-source",
        "--reliability-routing",
    ]
    main()
