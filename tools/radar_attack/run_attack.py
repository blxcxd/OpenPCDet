"""Canonical CLI entry point for 4D-radar attacks."""

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) in sys.path:
    sys.path.remove(str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from attacks.fgsm_attack_radar import main


if __name__ == '__main__':
    main()
