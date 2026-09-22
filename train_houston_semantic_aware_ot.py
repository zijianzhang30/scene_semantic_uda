"""Semantic-aware OT ablation: only the OT cost is changed."""
import sys
from train_houston_flow_transport import main

if __name__ == '__main__':
    sys.argv[1:1] = [
        '--scene-shift', '--scene-shift-aux-original-flow', '--fm-backprop-source',
        '--reliability-routing', '--routing-assignment', 'soft',
        '--semantic-aware-ot', '--lambda-sem', '0.1',
    ]
    main()
