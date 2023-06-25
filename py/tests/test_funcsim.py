import pytest
import numpy as np
from naiad.funcsim import (
    arc_length_array,
    array_to_function,
    derivative,
    FloatArray,
    FloatFn,
    xy_to_curvature,
)

EPS = 1e-2  # tolerance for floating point comparisons


@pytest.fixture
def exes() -> FloatArray:
    return np.linspace(0, 1, 1000)


@pytest.fixture
def wyes(exes: FloatArray) -> FloatArray:
    return np.power(exes, 2)


@pytest.fixture
def eses(exes: FloatArray, wyes: FloatArray) -> FloatArray:
    return arc_length_array(exes, wyes)


@pytest.fixture
def x_fn(eses: FloatArray, exes: FloatArray) -> FloatFn:
    return array_to_function(eses, exes)


@pytest.fixture
def y_fn(eses: FloatArray, wyes: FloatArray) -> FloatFn:
    return array_to_function(eses, wyes)


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


def test_xy_to_curvature(x_fn: FloatFn, y_fn: FloatFn):
    # curvature as a fn of s
    curvature_fn = xy_to_curvature(x_fn, y_fn)

    # curvature as a fn of x
    def true_curvature_fn(x: float) -> float:
        return 2.0 / (1 + 4 * x**2) ** 1.5

    # Make sure we are evaluating the two functions at the same point!
    s0 = 0.5
    x0 = x_fn(s0)

    assert curvature_fn(s0) == pytest.approx(true_curvature_fn(x0), EPS)
