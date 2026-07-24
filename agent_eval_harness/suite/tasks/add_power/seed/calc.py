"""A handful of pure functions with no third-party dependencies."""

from __future__ import annotations


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return a / b
