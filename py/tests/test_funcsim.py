import pytest
import numpy as np
from naiad.funcsim import (
    arc_length_array,
    array_to_function,
    derivative,
    FloatArray,
    FloatFn,
)

EPS = 1e-2  # tolerance for floating point comparisons


@pytest.fixture
def exes() -> FloatArray:
    return np.linspace(0, 1, 100)


@pytest.fixture
def wyes(exes: FloatArray) -> FloatArray:
    return np.power(exes, 2)


def test_arclength(exes: FloatArray, wyes: FloatArray):
    eses: FloatArray = arc_length_array(exes, wyes)

    def parabola_arclength(x: float) -> float:
        disc = np.sqrt(1 + 4 * x**2)
        return 0.5 * x * disc + 0.25 * np.log(2 * x + disc)

    true_eses = np.array([parabola_arclength(x) for x in exes])

    assert np.allclose(eses, true_eses, atol=EPS)


def test_array_to_function(exes: FloatArray, wyes: FloatArray):
    fn: FloatFn = array_to_function(exes, wyes)
    assert fn(0.5) == pytest.approx(0.25, EPS)


def test_derivative(exes: FloatArray, wyes: FloatArray):
    fn: FloatFn = array_to_function(exes, wyes)
    dfn: FloatFn = derivative(fn)
    assert dfn(0.5) == pytest.approx(1.0, EPS)
