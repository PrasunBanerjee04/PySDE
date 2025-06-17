import numpy as np
from utils import generate_time_grid


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    t0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    random_state=None
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion (GBM) via its exact solution:
        dS_t = mu * S_t dt + sigma * S_t dW_t
    """
    rng = np.random.default_rng(random_state)
    times = generate_time_grid(t0, T, n_steps)
    dt = (T - t0) / n_steps

    dW = rng.normal(0.0, np.sqrt(dt), size=(n_steps, n_paths))
    W = np.vstack([np.zeros(n_paths), np.cumsum(dW, axis=0)])

    exponent = (mu - 0.5 * sigma**2) * times[:, None] + sigma * W
    return S0 * np.exp(exponent)


def simulate_ou(
    X0: float,
    theta: float,
    mu: float,
    sigma: float,
    t0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    random_state=None
) -> np.ndarray:
    """
    Simulate Ornstein-Uhlenbeck (Vasicek) via exact discretization:
        dX_t = theta*(mu - X_t) dt + sigma dW_t
    """
    rng = np.random.default_rng(random_state)
    times = generate_time_grid(t0, T, n_steps)
    dt = (T - t0) / n_steps

    exp_term = np.exp(-theta * dt)
    var_term = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))

    X = np.zeros((n_steps + 1, n_paths))
    X[0, :] = X0

    for i in range(n_steps):
        mean = X[i, :] * exp_term + mu * (1 - exp_term)
        X[i+1, :] = mean + rng.normal(0.0, np.sqrt(var_term), size=n_paths)

    return X

#simulate_vasicek = simulate_ou


def simulate_cir(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    t0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    random_state=None
) -> np.ndarray:
    """
    Simulate Cox-Ingersoll-Ross (CIR) with full truncation Euler:
        dX_t = kappa*(theta - X_t) dt + sigma * sqrt(X_t) dW_t
    """
    rng = np.random.default_rng(random_state)
    dt = (T - t0) / n_steps

    X = np.zeros((n_steps + 1, n_paths))
    X[0, :] = X0

    for i in range(n_steps):
        X_prev = np.maximum(X[i, :], 0)
        dW = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        X[i+1, :] = (
            X_prev + kappa * (theta - X_prev) * dt
            + sigma * np.sqrt(X_prev) * dW
        )
    return X


def simulate_cev(
    S0: float,
    mu: float,
    sigma: float,
    gamma: float,
    t0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    random_state=None
) -> np.ndarray:
    """
    Simulate Constant Elasticity of Variance (CEV) via Euler-Maruyama:
        dS_t = mu * S_t dt + sigma * S_t^gamma dW_t
    """
    rng = np.random.default_rng(random_state)
    dt = (T - t0) / n_steps

    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    for i in range(n_steps):
        S_prev = S[i, :]
        dW = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        S[i+1, :] = (
            S_prev + mu * S_prev * dt
            + sigma * (S_prev ** gamma) * dW
        )
    return S


def simulate_heston(
    S0: float,
    V0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    random_state=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston model via Euler-Maruyama:
        dS_t = mu * S_t dt + sqrt(V_t) * S_t dW1_t
        dV_t = kappa*(theta - V_t) dt + xi * sqrt(V_t) dW2_t
    corr(dW1, dW2)=rho
    """
    rng = np.random.default_rng(random_state)
    dt = (T - t0) / n_steps

    S = np.zeros((n_steps + 1, n_paths))
    V = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0
    V[0, :] = V0

    for i in range(n_steps):
        dW1 = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * rng.normal(0.0, np.sqrt(dt), size=n_paths)
        V_prev = np.maximum(V[i, :], 0)
        S[i+1, :] = S[i, :] + mu * S[i, :] * dt + np.sqrt(V_prev) * S[i, :] * dW1
        V[i+1, :] = V_prev + kappa * (theta - V_prev) * dt + xi * np.sqrt(V_prev) * dW2
    return S, V


def simulate_merton_jump_diffusion(
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    t0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    random_state=None
) -> np.ndarray:
    """
    Simulate Merton jump-diffusion:
        dS_t = mu*S_t dt + sigma*S_t dW_t + S_t (J-1) dN_t
    where N is Poisson(lam) and jumps J=exp(Y), Y~N(mu_j, sigma_j^2).
    """
    rng = np.random.default_rng(random_state)
    dt = (T - t0) / n_steps

    # jump compensator
    kappa_j = np.exp(mu_j + 0.5 * sigma_j**2) - 1

    S = np.zeros((n_steps + 1, n_paths))
    S[0, :] = S0

    for i in range(n_steps):
        S_prev = S[i, :]
        dW = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        # diffusion + drift adjustment
        S_cont = S_prev * np.exp((mu - lam * kappa_j - 0.5 * sigma**2) * dt + sigma * dW)
        # jumps
        N = rng.poisson(lam * dt, size=n_paths)
        # total jump factor
        Y = rng.normal(loc=N * mu_j, scale=np.sqrt(N) * sigma_j)
        J = np.exp(Y)
        S[i+1, :] = S_cont * J
    return S


def simulate_sabr(
    F0: float,
    alpha0: float,
    beta: float,
    nu: float,
    rho: float,
    t0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    random_state=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate SABR model via Euler:
        dF = alpha * F^beta dW1
        dalpha = nu * alpha dW2, corr(dW1,dW2)=rho
    """
    rng = np.random.default_rng(random_state)
    dt = (T - t0) / n_steps

    F = np.zeros((n_steps + 1, n_paths))
    alpha = np.zeros((n_steps + 1, n_paths))
    F[0, :] = F0
    alpha[0, :] = alpha0

    for i in range(n_steps):
        dW1 = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * rng.normal(0.0, np.sqrt(dt), size=n_paths)
        F_prev = F[i, :]
        a_prev = alpha[i, :]
        F[i+1, :] = F_prev + a_prev * (F_prev ** beta) * dW1
        alpha[i+1, :] = a_prev + nu * a_prev * dW2
    return F, alpha
