"""
fin_sde.utils
-------------
Shared math and performance utilities for fin-sde.
"""

import numpy as np
from functools import wraps

try:
    from numba import jit as _numba_jit
except ImportError:
    _numba_jit = None


def make_time_grid(T, dt):
    """
    Create a 1D numpy array of time points from 0 to T (inclusive) with step dt.

    Parameters
    ----------
    T : float
        Total time horizon.
    dt : float
        Time step size.

    Returns
    -------
    numpy.ndarray
        Array of shape (N+1,) where N = int(T/dt).
    """
    N = int(np.ceil(T / dt))
    return np.linspace(0, N * dt, N + 1)


def finite_diff(f, x, h=1e-5):
    """
    Estimate the derivative of f at x via central finite differences.

    Parameters
    ----------
    f : callable
        Function of a single variable.
    x : float or array-like
        Point(s) at which to estimate the derivative.
    h : float
        Step size for finite differences.

    Returns
    -------
    float or numpy.ndarray
        Approximate derivative f'(x).
    """
    x = np.asarray(x)
    return (f(x + h) - f(x - h)) / (2 * h)


def normalize_series(series):
    """
    Z-score normalize a 1D array or pandas Series.

    Parameters
    ----------
    series : array-like or pandas.Series
        Input data.

    Returns
    -------
    numpy.ndarray
        Normalized array: (series - mean) / std.
    """
    arr = np.asarray(series, dtype=float)
    return (arr - np.nanmean(arr)) / np.nanstd(arr)


def jit(func=None, **kwargs):
    """
    Decorator to JIT-compile a function with numba if available.

    If numba is not installed, this decorator is a no-op.

    Usage:
        @jit(nopython=True, parallel=True)
        def f(x):
            ...

    Parameters
    ----------
    func : callable, optional
        The function to compile. If None, returns a decorator.
    **kwargs
        Keyword arguments passed to numba.jit.
    """
    if _numba_jit is None:
        # numba not available: return identity decorator
        def _identity(f):
            return f
        return _identity if func is None else func

    # numba is available
    if func is None:
        return lambda f: _numba_jit(**kwargs)(f)
    else:
        return _numba_jit(**kwargs)(func)


def safe_sqrt(x):
    """
    Compute the square root of x, ensuring non-negative inputs.

    Parameters
    ----------
    x : float or array-like
        Input values.

    Returns
    -------
    numpy.ndarray
        Elementwise sqrt(max(x, 0)).
    """
    arr = np.asarray(x)
    return np.sqrt(np.clip(arr, a_min=0, a_max=None))
