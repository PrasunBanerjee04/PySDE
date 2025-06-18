import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_price_series(
    series: np.ndarray,
    dt: float = 1.0,
    ax: Optional[plt.Axes] = None,
    title: str = "Price Series",
    xlabel: str = "Time",
    ylabel: str = "Price"
) -> plt.Axes:
    """
    Plot the raw price series against time.

    Parameters
    ----------
    series : 1D numpy array of prices.
    dt : float
        Time increment between observations (for x-axis scaling).
    ax : matplotlib Axes, optional
    title, xlabel, ylabel : labels for the plot.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    times = np.arange(series.size) * dt
    ax.plot(times, series, lw=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_simulation(
    times: np.ndarray,
    paths: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Simulated Paths",
    xlabel: str = "Time",
    ylabel: str = "Value",
    alpha: float = 0.6
) -> plt.Axes:
    """
    Plot multiple simulated paths.

    Parameters
    ----------
    times : 1D numpy array of time points.
    paths : 2D numpy array of shape (len(times), n_paths).
    ax : matplotlib Axes, optional
    alpha : float
        Transparency for individual paths.
    """
    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array of shape (n_steps, n_paths)")
    if ax is None:
        fig, ax = plt.subplots()
    # plot each path
    for j in range(paths.shape[1]):
        ax.plot(times, paths[:, j], alpha=alpha)
    # overlay mean path
    mean_path = paths.mean(axis=1)
    ax.plot(times, mean_path, 'k--', lw=2, label='Mean')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def plot_log_return_hist(
    series: np.ndarray,
    dt: float = 1.0,
    bins: int = 50,
    ax: Optional[plt.Axes] = None,
    title: str = "Log-Return Histogram",
    xlabel: str = "Log-Return",
    ylabel: str = "Frequency",
    overlay_normal: bool = True
) -> plt.Axes:
    """
    Plot histogram of log-returns with optional normal PDF overlay.
    """
    log_r = np.diff(np.log(series))
    if ax is None:
        fig, ax = plt.subplots()
    # histogram
    counts, edges, _ = ax.hist(log_r, bins=bins, density=True, alpha=0.6)

    if overlay_normal and log_r.size > 1:
        mu = np.mean(log_r)
        sigma = np.std(log_r, ddof=1)
        # PDF over range
        x = np.linspace(edges[0], edges[-1], 200)
        pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
        ax.plot(x, pdf, 'r-', lw=2, label=f'N($\mu$={mu:.3f}, $\sigma$={sigma:.3f})')
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_model_selection(
    results: Dict[str, Dict],
    criterion: str = 'aic',
    ax: Optional[plt.Axes] = None,
    title: str = "Model Selection",
    xlabel: str = "Model",
    ylabel: Optional[str] = None
) -> plt.Axes:
    """
    Bar plot of information criterion values for fitted models.

    Parameters
    ----------
    results : dict
        Keys are model names, values include 'aic' and 'bic'.
    criterion : {'aic', 'bic'}
    """
    criterion = criterion.lower()
    if criterion not in ('aic', 'bic'):
        raise ValueError("criterion must be 'aic' or 'bic'")
    models = list(results.keys())
    values = [results[m][criterion] for m in models]
    ylabel = ylabel or criterion.upper()

    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(models, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_ou_residuals(
    series: np.ndarray,
    dt: float,
    params: Dict[str, float],
    ax: Optional[plt.Axes] = None,
    title: str = "OU Residuals",
    xlabel: str = "X_{i}",
    ylabel: str = "Residual"
) -> plt.Axes:
    """
    Scatter plot of OU model residuals vs previous value.
    """
    theta = params['theta']
    mu = params['mu']
    phi = np.exp(-theta * dt)

    x0 = series[:-1]
    x1 = series[1:]
    resid = x1 - (phi * x0 + (1-phi) * mu)

    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x0, resid, alpha=0.6)
    ax.axhline(0, color='k', lw=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
