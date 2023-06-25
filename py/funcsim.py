from typing import Callable, Tuple
import numpy.typing as npt
import numpy as np
from scipy.interpolate import interp1d

EPS = 1e-3

RealFn = Callable[[float], float]


def arc_length_array(
    x_axis: npt.NDArray[np.float64], y_axis: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Get the arc length of a curve given by x_axis and y_axis.
    """
    assert len(x_axis) == len(y_axis)
    dx = np.diff(x_axis)
    dy = np.diff(y_axis)
    ss = np.cumsum(np.sqrt(dx**2 + dy**2))
    result = np.hstack([[0], ss])
    return result


def array_to_function(
    x_axis: npt.NDArray[np.float64], y_axis: npt.NDArray[np.float64]
) -> RealFn:
    return interp1d(x_axis, y_axis, kind="cubic", fill_value="extrapolate")


def derivative(fn: RealFn, eps: float = EPS) -> RealFn:
    def result(x: float, eps: float = eps) -> float:
        return (fn(x + eps) - fn(x - eps)) / (2 * eps)

    return result


def derivative_2(fn: RealFn, eps: float = EPS) -> RealFn:
    def result(x: float, eps: float = eps) -> float:
        return (fn(x + eps) - 2 * fn(x) + fn(x - eps)) / (eps**2)

    return result


def xy_to_curvature(x_fn: RealFn, y_fn: RealFn) -> RealFn:
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
        return numerator_fn(ss) / denominator

    return curvature_fn


def curvature_evolve(
    curvature_fn: RealFn, delta_t: float, alpha: float, vv: float
) -> RealFn:
    def result(ss: float) -> float:
        """
        Evolve curvature forward in time.
        Curvature is amplified and advected.
        """
        return np.exp(alpha * delta_t) * curvature_fn(ss - vv * delta_t)

    return result


def curvature_to_xy(curvature_fn: RealFn) -> Tuple[RealFn, RealFn]:
    theta_fn = integrate(curvature_fn)

    costheta_fn: RealFn = lambda ss: np.cos(theta_fn(ss))  # noqa: E731
    x_fn = integrate(costheta_fn)

    sintheta_fn: RealFn = lambda ss: np.sin(theta_fn(ss))  # noqa: E731
    y_fn = integrate(sintheta_fn)

    return x_fn, y_fn


def sample(fn: RealFn, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([fn(v) for v in values])


def integrate_strip(fn: RealFn, x0: float, x1: float) -> float:
    """
    Integrate a single strip using Simpson's rule
    """
    mid = (x0 + x1) / 2
    return (x1 - x0) / 6 * (fn(x0) + 4 * fn(mid) + fn(x1))


def integrate(fn: RealFn, x0: float = 0, eps: float = EPS) -> RealFn:
    """
    Given a real function and an initial value, return a function that
    takes a float x1 and computes the integral of the function from x0 to x1.
    """

    def integral(x1: float) -> float:
        strips: npt.NDArray[np.float64] = np.arange(x0, x1, eps)
        return np.sum(
            [
                integrate_strip(fn, strips[i], strips[i + 1])
                for i in range(len(strips) - 1)
            ]
        )

    return integral
