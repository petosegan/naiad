import pytest
import numpy as np
import numpy.typing as npt
from naiad.funcsim import (
    RealFn2,
    arc_length_array,
    array_to_function,
    derivative,
    xy_to_curvature,
)

FloatArray = npt.NDArray[np.float64]


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

    assert np.allclose(eses, true_eses, atol=1e-1)
