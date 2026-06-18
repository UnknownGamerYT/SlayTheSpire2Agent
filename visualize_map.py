from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from sts2sim.map_visualizer import main as visualizer_main

    return visualizer_main()


if __name__ == "__main__":
    raise SystemExit(main())
