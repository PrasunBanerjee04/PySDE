"""
PySDE: A Python package for simulating, estimating, and analyzing stochastic differential equations.
"""

__version__ = "0.1.0"

# Core simulation processes (renamed to simulate.py)
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

# Estimation routines
from .estimate import (
    estimate_gbm_mle,
    estimate_gbm_map,
    estimate_ou_mle,
    estimate_cir_mle,
    fit_sde,
    EstimationError
)

# Model selection & criteria
from .fit import (
    fit_models,
    select_model,
    ModelSelectionError,
    aic,
    bic
)

# Visualization
from .plot import (
    plot_price_series,
    plot_simulation,
    plot_log_return_hist,
    plot_model_selection,
    plot_ou_residuals
)

# Custom user-defined SDE interface
from .custom import CustomSDE

# Dataset utilities
from .dataset import (
    load_csv_series,
    load_multiple_csv,
    train_test_split_series,
    generate_synthetic_dataset,
    SDEDataset
)

# Utility functions
from .utils import (
    generate_time_grid,
    finite_diff,
    normalize_series,
    jit,
    safe_sqrt
)

__all__ = [
    "__version__",
    # simulation
    "generate_time_grid", "simulate_gbm", "simulate_ou", "simulate_vasicek",
    "simulate_cir", "simulate_cev", "simulate_heston",
    "simulate_merton_jump_diffusion", "simulate_sabr", "simulate_custom",
    # estimation
    "estimate_gbm_mle", "estimate_gbm_map", "estimate_ou_mle",
    "estimate_cir_mle", "fit_sde", "EstimationError",
    # model fitting
    "fit_models", "select_model", "ModelSelectionError", "aic", "bic",
    # plotting
    "plot_price_series", "plot_simulation", "plot_log_return_hist",
    "plot_model_selection", "plot_ou_residuals",
    # custom SDE
    "CustomSDE",
    # datasets
    "load_csv_series", "load_multiple_csv", "train_test_split_series",
    "generate_synthetic_dataset", "SDEDataset",
    # utilities
    "make_time_grid",
    "finite_diff", 
    "normalize_series",
    "jit",
    "safe_sqrt"
]
