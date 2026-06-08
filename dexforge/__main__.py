"""Allow ``python -m dexforge`` to invoke the CLI."""

from __future__ import annotations

from dexforge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
