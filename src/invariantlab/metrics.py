"""Pass-rate, Wilson interval and verification-gap metrics for repair runs."""

from __future__ import annotations

import math


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float] | None:
    """Return the Wilson score interval for a binomial proportion, or None if empty."""

    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) / total)
            + (z * z / (4.0 * total * total))
        )
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def pass_rate(successes: int, total: int) -> float | None:
    """Return ``successes / total``, or None if there are no attempts."""

    return successes / total if total else None


def verification_gap(
    public_rate: float | None,
    scientific_rate: float | None,
) -> float | None:
    """Return ``G = P_public - P_science``, or None if either rate is undefined."""

    if public_rate is None or scientific_rate is None:
        return None
    return public_rate - scientific_rate
