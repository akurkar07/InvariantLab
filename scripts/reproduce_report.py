"""Reproduce the smoke report.

Re-runs the reference evaluation and verifies outcomes match committed manifests.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce a report.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only.")
    args = parser.parse_args()

    # TODO: implement in M6
    print("Reproduce report: not yet implemented (coming in M6).")
    sys.exit(0)


if __name__ == "__main__":
    main()
