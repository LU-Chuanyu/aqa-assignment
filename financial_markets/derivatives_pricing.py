"""
Derivatives Pricing – AQA Assignment
======================================
Implements:
  - Black-Scholes model for European calls and puts
  - Option Greeks (Delta, Gamma, Vega, Theta, Rho)
  - Put-Call Parity check
  - Implied Volatility (numerical inversion via bisection)
"""

import numpy as np
import scipy.stats as stats


# ── 1. Black-Scholes pricing ──────────────────────────────────────────────────

def _d1_d2(
    S: float, K: float, r: float, sigma: float, T: float
) -> tuple[float, float]:
    """Compute d₁ and d₂ for the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes European call option price.

    Parameters
    ----------
    S     : current underlying price
    K     : strike price
    r     : risk-free interest rate (continuously compounded, annual)
    sigma : implied / historical volatility (annual)
    T     : time to expiry (years)

    Returns
    -------
    Call option price
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return float(S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))


def bs_put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes European put option price.

    Returns
    -------
    Put option price
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return float(
        K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    )


# ── 2. Option Greeks ──────────────────────────────────────────────────────────

def greeks(
    S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call"
) -> dict:
    """
    Compute Black-Scholes Greeks for a European option.

    Parameters
    ----------
    option_type : "call" or "put"

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, rho
    """
    if T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, r, sigma, T)
    n_d1 = stats.norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    gamma = float(n_d1 / (S * sigma * sqrt_T))
    vega = float(S * n_d1 * sqrt_T)           # per unit of σ (not per 1%)

    if option_type == "call":
        delta = float(stats.norm.cdf(d1))
        theta = float(
            -(S * n_d1 * sigma / (2 * sqrt_T))
            - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        )
        rho = float(K * T * np.exp(-r * T) * stats.norm.cdf(d2))
    else:
        delta = float(stats.norm.cdf(d1) - 1)
        theta = float(
            -(S * n_d1 * sigma / (2 * sqrt_T))
            + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
        )
        rho = float(-K * T * np.exp(-r * T) * stats.norm.cdf(-d2))

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ── 3. Put-Call Parity ────────────────────────────────────────────────────────

def put_call_parity_check(
    call: float,
    put: float,
    S: float,
    K: float,
    r: float,
    T: float,
    tol: float = 1e-6,
) -> bool:
    """
    Verify put-call parity: C − P = S − K e^{-rT}.

    Returns True if the relation holds within tolerance.
    """
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    return bool(abs(lhs - rhs) < tol)


# ── 4. Implied Volatility ─────────────────────────────────────────────────────

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 500,
) -> float:
    """
    Compute implied volatility by bisection (Newton inversion of B-S price).

    Returns
    -------
    Implied volatility (annual), or np.nan if no solution is found.
    """
    price_fn = bs_call_price if option_type == "call" else bs_put_price

    lo, hi = 1e-6, 10.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        mid_price = price_fn(S, K, r, mid, T)
        diff = mid_price - market_price
        if abs(diff) < tol:
            return float(mid)
        if diff > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break

    return float((lo + hi) / 2.0)
