import pytest

from calc import add, divide, multiply, subtract


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(4, 5) == 20


def test_divide():
    assert divide(10, 2) == 5
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
