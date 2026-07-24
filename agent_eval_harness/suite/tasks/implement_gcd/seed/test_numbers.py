from mathx import gcd


def test_gcd_basic():
    assert gcd(12, 8) == 4
    assert gcd(54, 24) == 6


def test_gcd_coprime():
    assert gcd(17, 5) == 1


def test_gcd_with_zero():
    assert gcd(0, 9) == 9
    assert gcd(9, 0) == 9
