import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless backend for plotting
import pytest

# core modules
from PySDE.simulate import (
    generate_time_grid,
    simulate_gbm,
    simulate_ou,
    simulate_cir,
    simulate_heston
)
from PySDE.custom import simulate_custom

# estimation
from PySDE.estimate import (
    estimate_gbm_mle,
    estimate_ou_mle,
    estimate_gbm_map,
    fit_sde,
    EstimationError
)

# model selection
from PySDE import fit as fit_module

# plotting
from PySDE import plot

# data loading / datasets
from PySDE import dataset
from PySDE.dataset import SDEDataset


def test_generate_time_grid():
    t0, T, n = 0.0, 1.0, 4
    grid = generate_time_grid(t0, T, n)
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (n+1,)
    assert grid[0] == pytest.approx(t0)
    assert grid[-1] == pytest.approx(T)


def test_simulate_gbm_deterministic():
    # sigma=0 -> deterministic exponential growth
    S0, mu = 100.0, 0.05
    paths = simulate_gbm(S0, mu, sigma=0.0, t0=0, T=1, n_steps=10, n_paths=3, random_state=42)
    times = generate_time_grid(0,1,10)
    expected = S0 * np.exp(mu * times)[:,None]
    assert np.allclose(paths, expected)


def test_simulate_ou_deterministic():
    # sigma=0 -> deterministic OU towards mu
    X0, theta, mu = 0.0, 2.0, 1.0
    paths = simulate_ou(X0, theta, mu, sigma=0.0, t0=0, T=1, n_steps=5, n_paths=2, random_state=1)
    times = generate_time_grid(0,1,5)
    exp_term = np.exp(-theta*(1/5))
    # check first step mean only
    assert np.allclose(paths[1], X0*exp_term + mu*(1-exp_term))


def test_simulate_cir_nonnegative():
    out = simulate_cir(0.1, 1.0, 0.2, 0.3, 0,1,10, n_paths=5, random_state=0)
    assert np.all(out >= 0)
    assert out.shape == (11,5)


def test_simulate_heston_shapes():
    S,V = simulate_heston(100,0.04,0.05,1.0,0.04,0.2,-0.5,0,1,10, n_paths=4, random_state=0)
    assert S.shape == (11,4)
    assert V.shape == (11,4)
    assert np.all(V >= 0)


def test_estimate_gbm_mle_and_map():
    # Use longer time series and more steps for better estimation
    S0, mu, sigma = 100.0, 0.1, 0.2
    times = generate_time_grid(0, 2, 2000)  # increased time period and steps
    data = simulate_gbm(S0, mu, sigma, 0, 2, 2000, n_paths=1, random_state=42).flatten()
    
    res_mle = estimate_gbm_mle(data, dt=2/2000)
    assert abs(res_mle['mu'] - mu) < 0.05
    assert abs(res_mle['sigma'] - sigma) < 0.05
    
    # MAP with informative prior
    priors = {'mu': (mu, 0.1)}  # tighter prior
    res_map = estimate_gbm_map(data, 2/2000, priors=priors)
    assert abs(res_map['mu'] - mu) < 0.05

def test_estimate_ou_mle():
    # simulate OU
    X0, theta, mu, sigma = 0.0, 1.5, 0.5, 0.1
    data = simulate_ou(X0, theta, mu, sigma, 0,1,100, n_paths=1, random_state=1).flatten()
    res = estimate_ou_mle(data, dt=1/100)
    assert abs(res['theta'] - theta) < 0.3
    assert abs(res['mu'] - mu) < 0.2
    assert hasattr(res, 'get')


def test_fit_sde_dispatch_errors():
    data = np.ones(10)
    with pytest.raises(ValueError):
        fit_sde(data, 'unknown', dt=1)
    with pytest.raises(ValueError):
        fit_sde(data, 'gbm', dt=-1)


def test_fit_models_and_select():
    # use constant series, only OU should fit
    data = np.ones(50)
    results = fit_module.fit_models(data, dt=1.0)
    assert 'ou' in results or 'gbm' in results
    model, metrics = fit_module.select_model(data, dt=1.0)
    assert isinstance(model, str)
    assert 'params' in metrics


def test_plot_functions_return_axes():
    series = np.linspace(1,2,10)
    ax1 = plot.plot_price_series(series)
    assert hasattr(ax1, 'plot')
    ax2 = plot.plot_log_return_hist(series)
    ax3 = plot.plot_model_selection({'gbm':{'aic':1,'bic':2}})
    # simulation plot
    times = np.arange(5)
    paths = np.tile(series[:5,None], (1,2))
    ax4 = plot.plot_simulation(times, paths)
    # ou residuals
    res = {'theta':1.0,'mu':1.0}
    ax5 = plot.plot_ou_residuals(series, dt=1.0, params=res)
    for ax in (ax2, ax3, ax4, ax5):
        assert hasattr(ax, 'set_title')


def test_custom_sde_simulation():
    # define simple linear drift/diffusion
    def drift(x,t): return -x
    def diffusion(x,t): return 0.0*x + 0.1
    X = simulate_custom(drift, diffusion, x0=1.0, t0=0, T=1, n_steps=10, n_paths=3, random_state=5)
    assert X.shape == (11,3)


def test_datasets_csv_and_synthetic(tmp_path):
    # create a temp CSV
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='D'),
        'price': np.arange(5.0)
    })
    file = tmp_path / 'test.csv'
    df.to_csv(file, index=False)
    times, vals = dataset.load_csv_series(str(file))
    assert len(times) == 5 and len(vals) == 5
    # multiple csv
    d = dataset.load_multiple_csv(str(tmp_path))
    assert 'test' in d
    # train-test split
    tr, te = dataset.train_test_split_series(np.arange(10), train_frac=0.6)
    assert len(tr) == 6 and len(te) == 4
    # synthetic
    times_syn, data_syn = dataset.generate_synthetic_dataset(
        'gbm', {'S0':10,'mu':0.1,'sigma':0.2}, 0,1,10, n_paths=2, random_state=0
    )
    assert times_syn.shape == (11,)
    assert data_syn.shape == (11,2)
    # SDEDataset wrapper
    ds = dataset.SDEDataset(times_syn, data_syn)
    train_ds, test_ds = ds.train_test_split(train_frac=0.5)
    df1 = ds.to_dataframe()
    assert 'value' in df1.columns or df1.shape[1] == 2