"""Command-line entry points for sdm-geometry."""

from __future__ import annotations

import argparse
import sys


def pilot_main(argv: list[str] | None = None) -> int:
    """Entry point for sdm-geometry-pilot command (placeholder).

    The real pilot logic lives in scripts/02_pilot_single_species.py
    so it stays easy to read and modify. This entry point exists
    primarily so `pip install -e .` exposes a console script we can
    later wire up.
    """
    parser = argparse.ArgumentParser(
        description="Run the local-ID × calibration pilot."
    )
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--species", default=None)
    parser.add_argument("--algorithm", default="rf")
    parser.add_argument("--track", default="full")
    parser.add_argument("--level", type=int, default=20)
    args = parser.parse_args(argv)

    if args.mode == "synthetic":
        from scripts import synthetic_runner  # type: ignore[import-not-found]

        return synthetic_runner.run()
    else:
        if args.species is None:
            print("--species required when --mode=real", file=sys.stderr)
            return 2
        from scripts import real_runner  # type: ignore[import-not-found]

        return real_runner.run(args.species, args.algorithm, args.track, args.level)


if __name__ == "__main__":
    sys.exit(pilot_main())
