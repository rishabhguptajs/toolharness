"""Classic fizzbuzz — with a deliberate bug for the fix task."""

from __future__ import annotations


def fizzbuzz(n: int) -> str:
    # BUG: the multiples-of-15 case must be checked first, otherwise multiples of
    # 15 are reported as "Fizz" instead of "FizzBuzz".
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    if n % 15 == 0:
        return "FizzBuzz"
    return str(n)
