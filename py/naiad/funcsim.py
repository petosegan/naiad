from typing import Callable, Tuple
import numpy.typing as npt
import numpy as np
from scipy.interpolate import interp1d  # type: ignore

EPS = 1e-3

FloatFn = Callable[[float], float]
FloatArray = npt.NDArray[np.float64]


class FloatFn2:
    """
    A function that takes a float and returns a float.

    Carries metadata to help with memoization
    """

    def __init__(self, fn: Callable[[float], float], functional_name: str = "UNK"):
        self.fn = fn
        self.functional_name = functional_name

    def __call__(self, x: float) -> float:
        return self.fn(x)

    def __str__(self) -> str:
        return f"{self.functional_name} ∘ {str(self.fn)}"


def arc_length_array(xcoords: FloatArray, ycoords: FloatArray) -> FloatArray:
    """
    Get the arc length, as an array, of a plane curve specified at xcoords and ycoords.
    """
    assert len(xcoords) == len(ycoords)
    dx = np.diff(xcoords)
    dy = np.diff(ycoords)
    ss = np.cumsum(np.sqrt(dx**2 + dy**2))
    result = np.hstack([[0], ss])  # add zero to make the array the same length
    return result


def array_to_function(x_array: FloatArray, y_array: FloatArray) -> FloatFn:
    """
    Given two arrays of floats, return a function that interpolates between them.

    The function computes y as a function of x.
    """
    interpolator: FloatFn = interp1d(
        x_array, y_array, kind="cubic", fill_value="extrapolate"
    )
    return interpolator


def derivative(fn: FloatFn, eps: float = EPS) -> FloatFn:
    """
    Given a function, return a function that computes the derivative of the function.
    """

    def result(x: float, eps: float = eps) -> float:
        return (fn(x + eps) - fn(x - eps)) / (2 * eps)

    return result


def xy_to_curvature(x_fn: FloatFn, y_fn: FloatFn) -> FloatFn:
    """
    Given two functions that compute x and y as a function of s,
    return a function that computes the curvature as a function of s.
    """
    dx_fn = derivative(x_fn)
    ddx_fn = derivative(dx_fn)
    dy_fn = derivative(y_fn)
    ddy_fn = derivative(dy_fn)

    def numerator_fn(ss: float) -> float:
        return dx_fn(ss) * ddy_fn(ss) - ddx_fn(ss) * dy_fn(ss)

    def denominator_fn(ss: float) -> float:
        return (dx_fn(ss) ** 2 + dy_fn(ss) ** 2) ** 1.5

    def curvature_fn(ss: float) -> float:
        denominator = denominator_fn(ss)
        if denominator == 0:
            return 0.0
        if ss < 0:
            return numerator_fn(0) / denominator
        return numerator_fn(ss) / denominator

    return curvature_fn


def curvature_evolve(
    curvature_fn: FloatFn, delta_t: float, alpha: float, vv: float
) -> FloatFn:
    """
    Given a function that computes curvature as a function of s at some initial time,
        and a timestamp delta_t,
    return a function that computes curvature as a function of s
        after delta_t has elapsed.
    """

    def result(ss: float) -> float:
        """
        Evolve curvature forward in time.
        Curvature is amplified and advected.
        """
        return np.exp(alpha * delta_t) * curvature_fn(ss - vv * delta_t)

    return result


def curvature_to_xy(
    curvature_fn: FloatFn, eps: float = 1e-2
) -> Tuple[FloatFn, FloatFn]:
    """
    Given a function that computes curvature as a function of s,
    return two functions that compute x and y as a function of s.
    """
    theta_fn = integrate(curvature_fn, 0, eps)

    costheta_fn: FloatFn = lambda ss: np.cos(theta_fn(ss))  # noqa: E731
    x_fn = integrate(costheta_fn, 0, eps)

    sintheta_fn: FloatFn = lambda ss: np.sin(theta_fn(ss))  # noqa: E731
    y_fn = integrate(sintheta_fn, 0, eps)

    return x_fn, y_fn


def sample(fn: FloatFn, values: FloatArray) -> FloatArray:
    """
    Given a function and an array of values,
    return an array of the function evaluated at those values.
    """
    return np.array([fn(v) for v in values])


def integrate_strip(fn: FloatFn, x0: float, x1: float) -> float:
    """
    Integrate a single strip using Simpson's rule
    """
    mid = (x0 + x1) / 2
    return (x1 - x0) / 6 * (fn(x0) + 4 * fn(mid) + fn(x1))


def integrate(fn: FloatFn, x0: float = 0, eps: float = EPS) -> FloatFn:
    """
    Given a real function and an initial value, return a function that
    takes a float x1 and computes the integral of the function from x0 to x1.
    """

    def integral(x1: float) -> float:
        strips: FloatArray = np.arange(x0, x1, eps)
        return np.sum(
            [
                integrate_strip(fn, strips[i], strips[i + 1])
                for i in range(len(strips) - 1)
            ]
        )

    return integral
