import pytest

from context import shapley

def v2(*args):
    if args == ():
        return 0.
    if args == (0,):
        return 8
    if args == (1,):
        return 1
    if args == (0, 1):
        return 10
    raise ValueError("Invalid inputs to characteristic function")

def v3(*args):
    if args == ():
        return 0
    if args == (0,):
        return 12
    if args == (1,):
        return 8
    if args == (2,):
        return 4
    if args == (0, 1):
        return 20
    if args == (0, 2):
        return 13
    if args == (1, 2):
        return 18
    if args == (0, 1, 2):
        return 19
    raise ValueError("Invalid inputs to characteristic function")


def test_shapley():

    # reference values calculated using http://shapleyvalue.com/index.php

    assert shapley.value(v2, {0,1}, 0) == pytest.approx(8.5)

    v3_sv = shapley.values(v3, {0,1,2})
    assert len(v3_sv) == 3
    assert v3_sv[0] == pytest.approx(7.833333)
    assert v3_sv[1] == pytest.approx(8.333333)
    assert v3_sv[2] == pytest.approx(2.833333)
