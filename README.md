# Asia Quant Academy (AQA) Assignment Repository

This repository contains assignments and review notes for the **Asia Quant Academy Global Quant Trading Program**. Each module covers a key topic in quantitative finance, with Python implementations and concise review summaries.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Setup](#setup)
3. [Module Overview & Review Points](#module-overview--review-points)
   - [1. Mathematics for Finance](#1-mathematics-for-finance)
   - [2. Python Programming](#2-python-programming)
   - [3. Financial Markets](#3-financial-markets)
   - [4. Quantitative Trading Strategies](#4-quantitative-trading-strategies)
   - [5. Risk Management](#5-risk-management)
   - [6. Machine Learning for Finance](#6-machine-learning-for-finance)

---

## Repository Structure

```
aqa-assignment/
├── README.md                        # This file – overview and review notes
├── requirements.txt                 # Python dependencies
├── mathematics/
│   ├── linear_algebra_review.py     # Vectors, matrices, eigenvalues, PCA
│   └── statistics_review.py        # Probability distributions, hypothesis tests
├── financial_markets/
│   ├── portfolio_theory.py          # MPT, efficient frontier, CAPM
│   └── derivatives_pricing.py      # Black-Scholes, Greeks
├── trading_strategies/
│   ├── backtester.py               # Event-driven backtesting framework
│   ├── moving_average.py           # Simple & exponential MA crossover
│   └── pairs_trading.py            # Statistical arbitrage / cointegration
├── risk_management/
│   ├── value_at_risk.py            # Historical, Parametric, Monte Carlo VaR
│   └── portfolio_risk.py           # Beta, tracking error, drawdown
└── machine_learning/
    ├── regression_models.py         # OLS, Ridge, Lasso for return prediction
    └── time_series.py              # ARIMA, autocorrelation, stationarity
```

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest --tb=short
```

---

## Module Overview & Review Points

### 1. Mathematics for Finance

**File:** `mathematics/`

#### Linear Algebra
| Concept | Key Points |
|---|---|
| Vectors & Dot Product | Used for portfolio weights and returns: `r_p = w · r` |
| Matrix Multiplication | Covariance matrix: `Σ = (1/n) Xᵀ X` |
| Eigenvalues/Eigenvectors | Principal Component Analysis (PCA) – dimension reduction |
| Cholesky Decomposition | Generating correlated random variables in simulation |
| Singular Value Decomposition (SVD) | Noise filtering of return matrices |

#### Probability & Statistics
| Concept | Key Points |
|---|---|
| Normal Distribution | Many finance models assume returns ~ N(μ, σ²) |
| Fat Tails / Kurtosis | Real returns have heavier tails than normal |
| Central Limit Theorem | Sum of i.i.d. returns converges to normal |
| Hypothesis Testing | t-test, p-value: used to validate alpha signals |
| Confidence Intervals | Range for parameter estimates; important in backtesting |
| Correlation vs. Causation | High correlation ≠ predictive relationship |
| MLE & OLS | Parameter estimation methods for models |

---

### 2. Python Programming

**Key Libraries:**

| Library | Use Case |
|---|---|
| `numpy` | Numerical arrays, linear algebra, random number generation |
| `pandas` | Time-series data, DataFrame manipulation |
| `scipy` | Statistics, optimization, interpolation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Machine learning algorithms |
| `statsmodels` | Statistical tests, OLS, ARIMA |
| `yfinance` | Download market data from Yahoo Finance |

**Review Points:**
- Use vectorized operations (NumPy/Pandas) – avoid Python `for` loops over large arrays
- `DataFrame.pct_change()` to compute log/simple returns
- Always align time series by index before operations
- `pd.concat`, `pd.merge` for combining datasets
- Use `rolling()` for moving-window calculations (MA, rolling volatility)

---

### 3. Financial Markets

**File:** `financial_markets/`

#### Modern Portfolio Theory (MPT)
| Concept | Formula | Key Points |
|---|---|---|
| Portfolio Return | `E[r_p] = Σ wᵢ E[rᵢ]` | Weighted average of expected returns |
| Portfolio Variance | `σ²_p = wᵀ Σ w` | Includes all pairwise correlations |
| Efficient Frontier | Minimize `σ²_p` for each target return | Pareto-optimal portfolios |
| Sharpe Ratio | `SR = (r_p − r_f) / σ_p` | Risk-adjusted return measure |
| Minimum Variance Portfolio | `w* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)` | Global minimum variance point |

#### CAPM
| Concept | Formula | Key Points |
|---|---|---|
| Security Market Line | `E[rᵢ] = r_f + βᵢ (E[r_m] − r_f)` | Expected return as linear function of β |
| Beta | `β = Cov(rᵢ, r_m) / Var(r_m)` | Systematic risk measure |
| Alpha | `α = r_p − [r_f + β(r_m − r_f)]` | Excess return above CAPM prediction |
| Jensen's Alpha | Regression intercept from `rᵢ − r_f = α + β(r_m − r_f)` | Statistical test of skill |

#### Derivatives Pricing
| Concept | Formula / Notes | Key Points |
|---|---|---|
| Black-Scholes (Call) | `C = S N(d₁) − K e^{-rT} N(d₂)` | Assumes constant σ, no dividends |
| d₁, d₂ | `d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)` | Moneyness adjusted for drift |
| Delta | `Δ = N(d₁)` (call) | Rate of change of option price w.r.t. S |
| Gamma | `Γ = N'(d₁) / (S σ √T)` | Rate of change of Delta |
| Vega | `ν = S N'(d₁) √T` | Sensitivity to volatility |
| Theta | `Θ = −[S N'(d₁) σ / (2√T)] − r K e^{-rT} N(d₂)` | Time decay |
| Put-Call Parity | `C − P = S − K e^{-rT}` | No-arbitrage relationship |
| Implied Volatility | Solve B-S numerically given observed market price | Volatility smile/skew |

---

### 4. Quantitative Trading Strategies

**File:** `trading_strategies/`

#### Backtesting Framework
| Concept | Key Points |
|---|---|
| Look-ahead Bias | Never use future data in signal generation |
| Transaction Costs | Include slippage + commissions; major impact on HFT |
| Survivorship Bias | Use point-in-time data; de-listed stocks matter |
| Overfitting | Walk-forward validation, out-of-sample testing |
| Annualization | Multiply returns by `√252` for daily Sharpe ratio |

#### Moving Average Crossover
- **Signal:** Buy when fast MA > slow MA; sell when fast MA < slow MA
- **Parameters:** Common pairs: (10, 50), (20, 200)
- **Review:** Trend-following; works in trending markets, fails in mean-reverting markets

#### Pairs Trading (Statistical Arbitrage)
| Step | Method |
|---|---|
| 1. Find cointegrated pair | Engle-Granger test; ADF test on spread |
| 2. Model spread | `spread = Pₐ − γ Pᵦ` where γ = hedge ratio (OLS) |
| 3. Generate signal | Z-score of spread: `z = (spread − μ) / σ` |
| 4. Trade | Long spread when z < −2; short spread when z > +2 |
| 5. Exit | Close when z returns to 0 |

**Review Points:**
- Cointegration ≠ correlation; cointegration implies long-run equilibrium
- Half-life of mean reversion: `HL = −ln(2) / β` from OU process fit
- Risk: regime changes can break cointegration

#### Performance Metrics
| Metric | Formula | Interpretation |
|---|---|---|
| Annualized Return | `(1 + r_total)^(252/N) − 1` | Compound annual growth rate |
| Sharpe Ratio | `mean(r) / std(r) × √252` | Return per unit of total risk |
| Sortino Ratio | `mean(r) / std(r_negative) × √252` | Penalises downside risk only |
| Max Drawdown | `(Peak − Trough) / Peak` | Largest peak-to-trough decline |
| Calmar Ratio | `Annual Return / Max Drawdown` | Return per unit of drawdown risk |
| Hit Rate | `# winning trades / # total trades` | Fraction of profitable trades |

---

### 5. Risk Management

**File:** `risk_management/`

#### Value at Risk (VaR)
| Method | Description | Formula |
|---|---|---|
| Historical | Sort past returns; take percentile | `VaR = −quantile(returns, 1−α)` |
| Parametric | Assume normal returns | `VaR = μ − z_α σ` |
| Monte Carlo | Simulate return paths | Average of worst `(1−α)%` simulated outcomes |

**Review Points:**
- VaR at 95% means: 5% chance of losing *at least* VaR amount in one period
- CVaR (Conditional VaR / Expected Shortfall) = expected loss *given* loss exceeds VaR
- VaR is **not** sub-additive; CVaR is a coherent risk measure
- Backtesting VaR: count VaR breaches; expect `α × T` breaches in `T` periods

#### Portfolio Risk Analytics
| Metric | Formula | Key Points |
|---|---|---|
| Portfolio Beta | `β_p = Σ wᵢ βᵢ` | Weighted sum of individual betas |
| Tracking Error | `std(r_p − r_benchmark)` | Deviation from benchmark |
| Information Ratio | `(r_p − r_b) / TE` | Active return per unit of tracking error |
| Max Drawdown | From equity curve | Critical for understanding tail risk |

---

### 6. Machine Learning for Finance

**File:** `machine_learning/`

#### Regression Models
| Model | Key Points |
|---|---|
| OLS (Linear Regression) | Baseline; interpretable; assumes linearity |
| Ridge (L2) | Penalises large coefficients; handles multicollinearity |
| Lasso (L1) | Feature selection via sparsity; useful for many predictors |
| Cross-Validation | K-fold CV; prevents overfitting |

**Feature Engineering:**
- Momentum: past N-day return
- Reversal: past 1-month return (short-term contrarian)
- Volatility: rolling standard deviation
- Volume signals: volume z-score, price-volume ratio
- Technical indicators: RSI, Bollinger Bands, MACD

#### Time Series Analysis
| Concept | Key Points |
|---|---|
| Stationarity | Required for most time-series models; test with ADF |
| ACF / PACF | Autocorrelation / Partial autocorrelation – model identification |
| ARIMA(p,d,q) | p = AR order, d = differencing, q = MA order |
| GARCH(p,q) | Models volatility clustering in financial returns |
| Cointegration | Long-run equilibrium between non-stationary series |

**Review Points:**
- Financial returns are generally stationary; prices are not
- GARCH useful for volatility forecasting and risk management
- ML models often need feature engineering; raw prices are not good features
- Out-of-sample performance is the only true test of predictive ability
- Beware of data snooping bias (p-hacking on many models/parameters)

---

## Quick Reference Cheat Sheet

```
Key Formulas
────────────────────────────────────────────────────────────────
Portfolio Return:      r_p = w · r
Portfolio Variance:    σ²_p = wᵀ Σ w
Sharpe Ratio:          SR = (r_p − r_f) / σ_p × √252
CAPM:                  E[r] = r_f + β (E[r_m] − r_f)
Black-Scholes Call:    C = S N(d₁) − K e^{-rT} N(d₂)
VaR (Parametric):      VaR = μ − z_α σ
Kelly Criterion:       f* = (bp − q) / b  where b=odds, p=win prob
Log Return:            r_log = ln(P_t / P_{t-1})
Simple Return:         r_simple = (P_t − P_{t-1}) / P_{t-1}
Annualize σ:           σ_annual = σ_daily × √252
────────────────────────────────────────────────────────────────
```
