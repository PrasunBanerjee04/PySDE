import numpy as np
from typing import List, Dict, Optional, Tuple

from .estimate import fit_sde
from .estimate import _nll_gbm


class ModelSelectionError(Exception):
    """
    Raised when no model can be fitted or selection fails.
    """
    pass


SUPPORTED_MODELS = ['gbm', 'ou', 'vasicek', 'cir']


def aic(nll: float, k: int) -> float:
    """
    Akaike Information Criterion: 2k + 2 * NLL
    """
    return 2 * k + 2 * nll


def bic(nll: float, k: int, n: int) -> float:
    """
    Bayesian Information Criterion: k*log(n) + 2 * NLL
    """
    return k * np.log(n) + 2 * nll


def _nll_ou(
    series: np.ndarray,
    dt: float,
    params: Dict[str, float]
) -> float:
    """
    Negative log-likelihood for discrete-time OU (AR(1)):
    X_{i+1} = phi*X_i + (1-phi)*mu + eps, eps ~ N(0, var_e)
    """
    theta = params['theta']
    mu = params['mu']
    # phi = exp(-theta dt)
    phi = np.exp(-theta * dt)

    x0 = series[:-1]
    x1 = series[1:]
    mean = phi * x0 + (1 - phi) * mu
    resid = x1 - mean
    # variance of residuals
    var_e = np.mean(resid**2)
    n = resid.size
    # population Gaussian NLL
    return 0.5 * (n * np.log(2 * np.pi * var_e) + np.sum(resid**2) / var_e)


def fit_models(
    series: np.ndarray,
    dt: float,
    models: Optional[List[str]] = None,
    method: str = 'mle'
) -> Dict[str, Dict[str, object]]:
    """
    Fit multiple candidate SDE models and compute fit metrics.

    Returns dict mapping model name to its params, NLL, AIC, BIC.
    """
    if not isinstance(series, np.ndarray) or series.ndim != 1:
        raise ValueError("series must be a one-dimensional numpy array.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    candidates = models or SUPPORTED_MODELS
    results: Dict[str, Dict[str, object]] = {}

    for model in candidates:
        model_lower = model.lower()
        if model_lower not in SUPPORTED_MODELS:
            continue
        try:
            # parameter estimation
            params = fit_sde(series, model_lower, dt, method=method)

            # compute negative log-likelihood
            if model_lower == 'gbm':
                # log-returns
                log_r = np.diff(np.log(series))
                nll = _nll_gbm(np.array([params['mu'], params['sigma']]), log_r, dt)
                k = 2
            elif model_lower in ('ou', 'vasicek'):
                nll = _nll_ou(series, dt, params)
                k = 3
            elif model_lower == 'cir':
                # CIR MLE not implemented yet
                continue
            else:
                continue

            # information criteria
            n = series.size
            results[model_lower] = {
                'params': params,
                'nll': nll,
                'aic': aic(nll, k),
                'bic': bic(nll, k, n)
            }
        except Exception:
            # skip models that fail
            continue

    if not results:
        raise ModelSelectionError("No models could be fitted to the data.")
    return results


def select_model(
    series: np.ndarray,
    dt: float,
    models: Optional[List[str]] = None,
    method: str = 'mle',
    criterion: str = 'aic'
) -> Tuple[str, Dict[str, object]]:
    """
    Select the best model according to aic or bic.

    Returns (best_model_name, metrics_dict).
    """
    criterion = criterion.lower()
    if criterion not in ('aic', 'bic'):
        raise ValueError("criterion must be 'aic' or 'bic'.")

    fits = fit_models(series, dt, models=models, method=method)

    # pick the model with lowest criterion value
    best = min(
        fits.items(),
        key=lambda item: item[1][criterion]
    )
    return best  # (model_name, metrics)
