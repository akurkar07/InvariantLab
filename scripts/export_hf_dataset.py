"""Export evaluation results to a Hugging Face dataset.

Produces a dataset with task metadata, trajectories, patches, measurements and provenance.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export to Hugging Face dataset.")
    parser.add_argument("--output", required=True, help="Output path or HF repo ID.")
    args = parser.parse_args()

    # TODO: implement in M6
    print(f"Export to HF dataset: not yet implemented (coming in M6).")


if __name__ == "__main__":
    main()
