"""Backward-compatible CLI for the modular 4D-radar attack runner."""

import os
import sys

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TOOLS_DIR in sys.path:
    sys.path.remove(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from radar_attack.runner import main


if __name__ == '__main__':
    main()
