import os
import glob
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd

from .simulate import (
    generate_time_grid,
    simulate_gbm,
    simulate_ou,
    simulate_cir,
    simulate_cev,
    simulate_heston,
    simulate_merton_jump_diffusion,
    simulate_sabr
)
from .custom import simulate_custom


def load_csv_series(
    filepath: str,
    date_col: str = 'date',
    value_col: str = 'price',
    parse_dates: bool = True,
    dropna: bool = True
) -> Tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
    """
    Load a single-column time series from CSV.

    Parameters
    ----------
    filepath : path to CSV file
    date_col : name of the date/time column
    value_col : name of the series values column
    parse_dates : whether to parse date_col as datetime
    dropna : drop missing values

    Returns
    -------
    tuple(times, values)
      - times: np.ndarray of datetime64 if parse_dates else float index
      - values: np.ndarray of floats
    """
    df = pd.read_csv(filepath, parse_dates=[date_col] if parse_dates else None)
    if dropna:
        df = df.dropna(subset=[date_col, value_col])
    times = df[date_col].values if parse_dates else df.index.values.astype(float)
    values = df[value_col].values.astype(float)
    return times, values


def load_multiple_csv(
    directory: str,
    pattern: str = '*.csv',
    date_col: str = 'date',
    value_col: str = 'price'
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all CSV files matching pattern into a dict of series.

    Returns
    -------
    dict where keys are filenames (without extension) and values are (times, values)
    """
    files = glob.glob(os.path.join(directory, pattern))
    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        times, values = load_csv_series(path, date_col=date_col, value_col=value_col)
        data[name] = (times, values)
    return data


def train_test_split_series(
    series: np.ndarray,
    train_frac: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a 1D numpy array into train and test segments.

    Parameters
    ----------
    series : 1D array
    train_frac : fraction of data to use for training

    Returns
    -------
    (train_series, test_series)
    """
    if series.ndim != 1:
        raise ValueError("series must be one-dimensional")
    n = series.size
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must be in (0,1)")
    split = int(np.floor(n * train_frac))
    return series[:split], series[split:]


def generate_synthetic_dataset(
    model: str,
    params: Dict,
    t0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    random_state=None,
    custom_funcs: Optional[Dict[str, callable]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic SDE data for a given model.

    Parameters
    ----------
    model : one of 'gbm','ou','cir','cev','heston','merton_jump','sabr','custom'
    params : parameters dict for the simulator
    custom_funcs : for model='custom', pass {'drift':..., 'diffusion':..., 'diffusion_derivative':...}

    Returns
    -------
    (times, data) where data is np.ndarray shape (n_steps+1,) for n_paths=1 
    or (n_steps+1, n_paths) for n_paths>1 or tuple for multivariate models
    """
    model = model.lower()
    times = generate_time_grid(t0, T, n_steps)
    
    if model == 'gbm':
        data = simulate_gbm(
            S0=params['S0'], mu=params['mu'], sigma=params['sigma'],
            t0=t0, T=T, n_steps=n_steps, n_paths=n_paths, random_state=random_state
        )
        if n_paths == 1:
            data = data.flatten()
            
    elif model in ('ou', 'vasicek'):
        data = simulate_ou(
            X0=params['X0'], theta=params['theta'], mu=params['mu'], sigma=params['sigma'],
            t0=t0, T=T, n_steps=n_steps, n_paths=n_paths, random_state=random_state
        )
        if n_paths == 1:
            data = data.flatten()
            
    elif model == 'cir':
        data = simulate_cir(
            X0=params['X0'], kappa=params['kappa'], theta=params['theta'], sigma=params['sigma'],
            t0=t0, T=T, n_steps=n_steps, n_paths=n_paths, random_state=random_state
        )
        if n_paths == 1:
            data = data.flatten()
            
    elif model == 'cev':
        data = simulate_cev(
            S0=params['S0'], mu=params['mu'], sigma=params['sigma'], gamma=params['gamma'],
            t0=t0, T=T, n_steps=n_steps, n_paths=n_paths, random_state=random_state
        )
        if n_paths == 1:
            data = data.flatten()
            
    elif model == 'heston':
        data = simulate_heston(
            S0=params['S0'], V0=params['V0'], mu=params['mu'], kappa=params['kappa'],
            theta=params['theta'], xi=params['xi'], rho=params['rho'],
            t0=t0, T=T, n_steps=n_steps, n_paths=n_paths, random_state=random_state
        )
        # Heston returns a tuple (S, V), don't flatten
            
    elif model == 'merton_jump':
        data = simulate_merton_jump_diffusion(
            S0=params['S0'], mu=params['mu'], sigma=params['sigma'], lam=params['lam'],
            mu_j=params['mu_j'], sigma_j=params['sigma_j'],
            t0=t0, T=T, n_steps=n_steps, n_paths=n_paths, random_state=random_state
        )
        if n_paths == 1:
            data = data.flatten()
            
    elif model == 'sabr':
        data = simulate_sabr(
            F0=params['F0'], alpha0=params['alpha0'], beta=params['beta'],
            nu=params['nu'], rho=params['rho'],
            t0=t0, T=T, n_steps=n_steps, n_paths=n_paths, random_state=random_state
        )
        # SABR returns a tuple (F, alpha), don't flatten
            
    elif model == 'custom':
        if not custom_funcs or 'drift' not in custom_funcs or 'diffusion' not in custom_funcs:
            raise ValueError("Must provide 'drift' and 'diffusion' for custom model")
        data = simulate_custom(
            drift=custom_funcs['drift'], diffusion=custom_funcs['diffusion'],
            diffusion_derivative=custom_funcs.get('diffusion_derivative', None),
            x0=params.get('x0'), t0=t0, T=T,
            n_steps=n_steps, n_paths=n_paths, method=params.get('method','euler'), 
            random_state=random_state
        )
        if n_paths == 1:
            data = data.flatten()
            
    else:
        raise ValueError(f"Unknown model '{model}'.")
        
    return times, data


class SDEDataset:
    """
    Wrapper for time-series datasets from CSV or synthetic SDE generation.
    """
    def __init__(
        self,
        times: np.ndarray,
        data: np.ndarray
    ):
        if times.ndim != 1:
            raise ValueError("times must be a 1D array")
        # data can be 2D (univariate) or tuple for multivariate
        self.times = times
        self.data = data

    def train_test_split(
        self,
        train_frac: float = 0.7
    ) -> Tuple['SDEDataset', 'SDEDataset']:
        """
        Split the data into train and test SDEDatasets.
        """
        if isinstance(self.data, tuple):
            # multivariate: split each
            split = int(np.floor(self.times.size * train_frac))
            t_train = self.times[:split]
            t_test = self.times[split:]
            data_train = tuple(d[:split] for d in self.data)
            data_test = tuple(d[split:] for d in self.data)
        else:
            series = self.data
            split = int(np.floor(series.shape[0] * train_frac))
            t_train = self.times[:split]
            t_test = self.times[split:]
            data_train = series[:split]
            data_test = series[split:]

        return (
            SDEDataset(t_train, data_train),
            SDEDataset(t_test, data_test)
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert times and data into a pandas DataFrame.
        """
        if isinstance(self.data, tuple):
            cols: Dict[str, np.ndarray] = {}
            for i, arr in enumerate(self.data):
                cols[f'var{i}'] = arr
            df = pd.DataFrame(cols, index=self.times)
        else:
            df = pd.DataFrame({'value': self.data}, index=self.times)
        df.index.name = 'time'
        return df
