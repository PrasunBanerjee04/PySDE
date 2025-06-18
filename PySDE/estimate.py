import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional, Any


class EstimationError(Exception):
    """
    Custom exception for estimation failures.
    """
    pass


def _validate_series(data: np.ndarray):
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 1:
        raise ValueError("Input series must be one-dimensional.")
    if data.size < 2:
        raise ValueError("At least two data points are required for estimation.")


def estimate_gbm_mle(
    prices: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """
    MLE for GBM parameters based on log-returns.

    Parameters
    ----------
    prices : 1D array of asset prices.
    dt : time increment between observations.

    Returns
    -------
    dict with keys 'mu' and 'sigma'.
    """
    _validate_series(prices)
    if dt <= 0:
        raise ValueError("dt must be positive.")

    log_r = np.diff(np.log(prices))
    n = log_r.size

    mean_lr = np.mean(log_r)
    var_lr = np.var(log_r, ddof=1)

    sigma = np.sqrt(var_lr / dt)
    mu = mean_lr / dt + 0.5 * sigma**2
    return {"mu": mu, "sigma": sigma}


def estimate_ou_mle(
    series: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """
    MLE for Ornstein–Uhlenbeck (Vasicek) parameters.

    Model: dX_t = theta*(mu - X_t) dt + sigma dW_t
    Discrete: X_{i+1} = phi X_i + (1-phi) mu + eps,  phi = exp(-theta dt)

    Returns 'theta', 'mu', 'sigma'.
    """
    _validate_series(series)
    if dt <= 0:
        raise ValueError("dt must be positive.")

    x0 = series[:-1]
    x1 = series[1:]
    n = x0.size

    # Calculate phi with numerical stability check
    denominator = np.sum(x0 * x0)
    if denominator < 1e-10:  # Avoid division by zero
        raise EstimationError("Numerically unstable estimation - near constant series")
    
    phi = np.sum(x0 * x1) / denominator
    
    # Clip phi to valid range with small buffer
    phi = np.clip(phi, 1e-6, 1 - 1e-6)
    
    # Calculate remaining parameters
    theta = -np.log(phi) / dt
    mu = (np.mean(x1) - phi * np.mean(x0)) / (1 - phi)

    # Calculate sigma with stability checks
    resid = x1 - (phi * x0 + (1 - phi) * mu)
    var_e = np.sum(resid**2) / n
    sigma = np.sqrt(np.maximum(2 * theta * var_e / (1 - phi**2), 1e-10))

    return {"theta": theta, "mu": mu, "sigma": sigma}

def estimate_cir_mle(
    series: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """
    Placeholder for CIR parameter estimation.
    Currently not implemented.
    """
    raise NotImplementedError("CIR MLE is not yet implemented.")


def _nll_gbm(params: np.ndarray, log_r: np.ndarray, dt: float) -> float:
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    n = log_r.size
    ll = -(-n/2 * np.log(2*np.pi*sigma**2*dt)
           - np.sum((log_r - (mu - 0.5*sigma**2)*dt)**2) / (2*sigma**2*dt))
    return ll


def estimate_gbm_map(
    prices: np.ndarray,
    dt: float,
    priors: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    MAP estimation for GBM. Priors can include normal/prior specs for mu and sigma.
    Uses scipy.optimize.minimize.
    """
    _validate_series(prices)
    log_r = np.diff(np.log(prices))

    init = np.array([0.0, 0.1])

    def neg_log_posterior(params):
        nll = _nll_gbm(params, log_r, dt)
        if priors:
            for key, prior in priors.items():
                if key == 'mu':
                    m, v = prior
                    nll += 0.5*((params[0]-m)**2)/v
                elif key == 'sigma':
                    pass
        return nll

    res = minimize(
        neg_log_posterior,
        x0=init,
        bounds=[(None, None), (1e-6, None)]
    )
    if not res.success:
        raise EstimationError("GBM MAP optimization failed: " + res.message)
    mu_hat, sigma_hat = res.x
    return {"mu": mu_hat, "sigma": sigma_hat}


def fit_sde(
    series: np.ndarray,
    model: str,
    dt: float,
    method: str = 'mle',
    priors: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Dispatch function to estimate SDE parameters.

    Parameters
    ----------
    series : data series of the underlying process.
    model : one of 'gbm', 'ou', 'cir'.
    dt : time step.
    method : 'mle' or 'map'.
    priors : prior specs for 'map'.
    """
    _validate_series(series)
    if method not in ('mle', 'map'):
        raise ValueError("method must be 'mle' or 'map'.")

    model = model.lower()
    if model == 'gbm':
        if method == 'mle':
            return estimate_gbm_mle(series, dt)
        else:
            return estimate_gbm_map(series, dt, priors)
    elif model in ('ou', 'vasicek'):
        if method != 'mle':
            raise NotImplementedError("MAP for OU not implemented.")
        return estimate_ou_mle(series, dt)
    elif model == 'cir':
        return estimate_cir_mle(series, dt)
    else:
        raise ValueError(f"Unknown model '{model}'. Supported: gbm, ou, cir.")
