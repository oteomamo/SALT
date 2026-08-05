# -*- coding: utf-8 -*-
"""The salt-mcp entry point.

Argument parsing and the version handshake live here; the tools arrive
over the 2.10.z line. Imports stay light on purpose: this module is what
a client execs, and a server that loads torch before it has been asked
for anything is a server that looks broken while it starts.
"""

import argparse
import sys


def salt_version():
    """The installed version, or the source tree's when running from a
    checkout that was never installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version("salt")
    except Exception:
        pass
    import re
    from pathlib import Path
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        match = re.search(r'^version = "([^"]+)"',
                          pyproject.read_text(encoding="utf-8"), re.M)
    except OSError:
        return "unknown"
    return match.group(1) if match else "unknown"


def build_parser():
    p = argparse.ArgumentParser(
        prog="salt-mcp", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="store_true",
                   help="print the salt version this server carries")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.version:
        print(salt_version())
        return 0
    print("salt-mcp has no tools yet - the server arrives over the 2.10.z "
          "line.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
