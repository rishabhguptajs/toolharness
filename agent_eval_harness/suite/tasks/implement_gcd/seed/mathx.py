"""Number-theory helpers."""

from __future__ import annotations


def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of two non-negative integers."""
    raise NotImplementedError("implement me")


def lcm(a: int, b: int) -> int:
    """Least common multiple, defined in terms of gcd."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)
