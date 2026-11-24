"""Launcher: the code lives in the `contextgate_core` package. This file keeps `python mcp_server.py` and the absolute script paths
registered by `smarty init` / the VS Code extension working from a clone."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextgate_core.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
