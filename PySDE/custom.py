import numpy as np
from typing import Callable, Optional, Tuple
from .simulate import generate_time_grid


class CustomSDE:
    """
    Define and simulate a custom Itô SDE:
        dX_t = drift(X_t, t) dt + diffusion(X_t, t) dW_t
    Optionally, for 1D processes, provide diffusion_derivative for Milstein.
    """
    def __init__(
        self,
        drift: Callable[[np.ndarray, float], np.ndarray],
        diffusion: Callable[[np.ndarray, float], np.ndarray],
        diffusion_derivative: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    ):
        if not callable(drift) or not callable(diffusion):
            raise TypeError("drift and diffusion must be callables")
        self.drift = drift
        self.diffusion = diffusion
        self.diffusion_derivative = diffusion_derivative

    def simulate(
        self,
        x0: np.ndarray,
        t0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        method: str = "euler",
        random_state=None
    ) -> np.ndarray:
        """
        Simulate the SDE using specified scheme.

        Parameters
        ----------
        x0 : array-like
            Initial state (scalar or vector).
        t0 : float
            Initial time.
        T : float
            Terminal time.
        n_steps : int
            Number of steps.
        n_paths : int
            Number of Monte Carlo paths.
        method : {'euler', 'milstein'}
            Discretization scheme.
        random_state : int or Generator
            RNG seed for reproducibility.

        Returns
        -------
        X : np.ndarray
            Array of shape (n_steps+1, *state_shape, n_paths).
        """
        method = method.lower()
        if method not in ("euler", "milstein"):
            raise ValueError("method must be 'euler' or 'milstein'")
        if method == "milstein" and self.diffusion_derivative is None:
            raise AttributeError(
                "diffusion_derivative must be provided for Milstein scheme"
            )
        return (self._simulate_euler(
            x0, t0, T, n_steps, n_paths, random_state
        ) if method == "euler" else
                self._simulate_milstein(
            x0, t0, T, n_steps, n_paths, random_state
        ))

    def _simulate_euler(
        self,
        x0,
        t0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        random_state
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        times = generate_time_grid(t0, T, n_steps)
        dt = (T - t0) / n_steps

        x0_arr = np.array(x0)
        state_shape = x0_arr.shape
        # Allocate array: time × state_shape × paths
        X = np.zeros((n_steps + 1, *state_shape, n_paths))
        X[0, ..., :] = np.expand_dims(x0_arr, axis=-1)

        for i in range(n_steps):
            t = times[i]
            x_curr = X[i]
            drift_term = self.drift(x_curr, t) * dt
            dW = rng.normal(0.0, np.sqrt(dt), size=(*state_shape, n_paths))
            diffusion_term = self.diffusion(x_curr, t) * dW
            X[i + 1] = x_curr + drift_term + diffusion_term
        return X

    def _simulate_milstein(
        self,
        x0,
        t0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        random_state
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        times = generate_time_grid(t0, T, n_steps)
        dt = (T - t0) / n_steps

        # Only supports scalar SDE (state_shape=())
        X = np.zeros((n_steps + 1, n_paths))
        X[0, :] = x0

        for i in range(n_steps):
            t = times[i]
            x_prev = X[i, :]
            f = self.drift(x_prev, t)
            g = self.diffusion(x_prev, t)
            g_x = self.diffusion_derivative(x_prev, t)
            dW = rng.normal(0.0, np.sqrt(dt), size=n_paths)
            X[i + 1] = (
                x_prev + f * dt
                + g * dW
                + 0.5 * g * g_x * (dW**2 - dt)
            )
        return X


def simulate_custom(
    drift: Callable[[np.ndarray, float], np.ndarray],
    diffusion: Callable[[np.ndarray, float], np.ndarray],
    x0,
    t0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    method: str = "euler",
    diffusion_derivative: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    random_state=None
) -> np.ndarray:
    """
    Convenience function: define and simulate a custom SDE in one call.
    """
    sde = CustomSDE(drift, diffusion, diffusion_derivative)
    return sde.simulate(x0, t0, T, n_steps, n_paths, method, random_state)
