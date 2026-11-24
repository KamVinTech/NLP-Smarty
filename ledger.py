"""Launcher: the code lives in the `contextgate` package. This file keeps `python ledger.py` and the absolute script paths
registered by `smarty init` / the VS Code extension working from a clone."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextgate.ledger import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
