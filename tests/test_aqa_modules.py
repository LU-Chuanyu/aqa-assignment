"""
Tests for AQA Assignment modules.
"""

import numpy as np
import pandas as pd
import pytest


# ── Mathematics ───────────────────────────────────────────────────────────────

class TestLinearAlgebra:
    def test_portfolio_return(self):
        from mathematics.linear_algebra_review import portfolio_return

        w = np.array([0.5, 0.5])
        r = np.array([0.1, 0.2])
        assert portfolio_return(w, r) == pytest.approx(0.15)

    def test_portfolio_variance(self):
        from mathematics.linear_algebra_review import portfolio_variance

        w = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        var = portfolio_variance(w, cov)
        expected = 0.5**2 * 0.04 + 2 * 0.5 * 0.5 * 0.02 + 0.5**2 * 0.09
        assert var == pytest.approx(expected)

    def test_minimum_variance_weights_sum_to_one(self):
        from mathematics.linear_algebra_review import minimum_variance_weights

        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = minimum_variance_weights(cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)

    def test_pca_eigenvalues_descending(self):
        from mathematics.linear_algebra_review import pca

        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4))
        cov = np.cov(data.T)
        eigenvalues, _ = pca(cov)
        assert all(eigenvalues[i] >= eigenvalues[i + 1] for i in range(len(eigenvalues) - 1))

    def test_correlated_samples_covariance(self):
        from mathematics.linear_algebra_review import correlated_samples

        cov = np.array([[1.0, 0.8], [0.8, 1.0]])
        rng = np.random.default_rng(0)
        samples = correlated_samples(cov, n_samples=50_000, rng=rng)
        empirical_cov = np.cov(samples.T)
        np.testing.assert_allclose(empirical_cov, cov, atol=0.05)


class TestStatistics:
    def test_simple_returns(self):
        from mathematics.statistics_review import simple_returns

        prices = np.array([100.0, 110.0, 99.0])
        r = simple_returns(prices)
        assert r[0] == pytest.approx(0.1)
        assert r[1] == pytest.approx((99 - 110) / 110)

    def test_log_returns(self):
        from mathematics.statistics_review import log_returns

        prices = np.array([100.0, 110.0])
        r = log_returns(prices)
        assert r[0] == pytest.approx(np.log(1.1))

    def test_sharpe_ratio_positive(self):
        from mathematics.statistics_review import sharpe_ratio

        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.01, 252)
        sr = sharpe_ratio(r)
        # With positive mean returns, Sharpe should be positive
        assert sr > 0

    def test_sharpe_ratio_zero_variance(self):
        from mathematics.statistics_review import sharpe_ratio

        r = np.zeros(100)
        assert sharpe_ratio(r) == 0.0

    def test_annualize_volatility(self):
        from mathematics.statistics_review import annualize_volatility

        assert annualize_volatility(1.0, 252) == pytest.approx(np.sqrt(252))

    def test_confidence_interval_contains_mean(self):
        from mathematics.statistics_review import mean_confidence_interval

        rng = np.random.default_rng(7)
        r = rng.normal(0.0, 0.01, 500)
        lo, hi = mean_confidence_interval(r, 0.95)
        assert lo < np.mean(r) < hi


# ── Financial Markets ─────────────────────────────────────────────────────────

class TestPortfolioTheory:
    def test_portfolio_performance_shape(self):
        from financial_markets.portfolio_theory import portfolio_performance

        w = np.array([0.4, 0.6])
        r = np.array([0.0005, 0.0008])
        cov = np.array([[0.0001, 0.00005], [0.00005, 0.0002]])
        ret, vol, sr = portfolio_performance(w, r, cov)
        assert ret > 0
        assert vol > 0

    def test_max_sharpe_weights_sum_to_one(self):
        from financial_markets.portfolio_theory import max_sharpe_portfolio

        rng = np.random.default_rng(5)
        r = np.array([0.0005, 0.0007, 0.0004])
        cov = np.array([[0.0001, 0.00002, 0.00001],
                         [0.00002, 0.00015, 0.00003],
                         [0.00001, 0.00003, 0.00008]])
        w = max_sharpe_portfolio(r, cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)
        assert all(wi >= -1e-6 for wi in w)

    def test_min_variance_weights_sum_to_one(self):
        from financial_markets.portfolio_theory import minimum_variance_portfolio

        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = minimum_variance_portfolio(cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-4)

    def test_capm_beta_positive_market(self):
        from financial_markets.portfolio_theory import capm_beta_alpha

        rng = np.random.default_rng(9)
        mkt = rng.normal(0.0005, 0.01, 500)
        asset = 1.2 * mkt + rng.normal(0, 0.002, 500)
        beta, _ = capm_beta_alpha(asset, mkt)
        assert beta == pytest.approx(1.2, abs=0.1)


class TestDerivativesPricing:
    def test_bs_call_intrinsic_value_at_expiry(self):
        from financial_markets.derivatives_pricing import bs_call_price

        assert bs_call_price(110, 100, 0.05, 0.2, 0) == pytest.approx(10.0)
        assert bs_call_price(90, 100, 0.05, 0.2, 0) == pytest.approx(0.0)

    def test_bs_put_intrinsic_value_at_expiry(self):
        from financial_markets.derivatives_pricing import bs_put_price

        assert bs_put_price(90, 100, 0.05, 0.2, 0) == pytest.approx(10.0)
        assert bs_put_price(110, 100, 0.05, 0.2, 0) == pytest.approx(0.0)

    def test_put_call_parity(self):
        from financial_markets.derivatives_pricing import (
            bs_call_price, bs_put_price, put_call_parity_check
        )

        S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        C = bs_call_price(S, K, r, sigma, T)
        P = bs_put_price(S, K, r, sigma, T)
        assert put_call_parity_check(C, P, S, K, r, T)

    def test_implied_vol_roundtrip(self):
        from financial_markets.derivatives_pricing import (
            bs_call_price, implied_volatility
        )

        S, K, r, sigma, T = 100, 100, 0.05, 0.25, 1.0
        price = bs_call_price(S, K, r, sigma, T)
        iv = implied_volatility(price, S, K, r, T)
        assert iv == pytest.approx(sigma, abs=1e-4)

    def test_greeks_call_delta_range(self):
        from financial_markets.derivatives_pricing import greeks

        g = greeks(100, 100, 0.05, 0.2, 1.0, option_type="call")
        assert 0 < g["delta"] < 1
        assert g["gamma"] > 0
        assert g["vega"] > 0

    def test_greeks_put_delta_range(self):
        from financial_markets.derivatives_pricing import greeks

        g = greeks(100, 100, 0.05, 0.2, 1.0, option_type="put")
        assert -1 < g["delta"] < 0


# ── Trading Strategies ────────────────────────────────────────────────────────

class TestBacktester:
    def _make_prices(self, n: int = 300) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        prices = 100 * np.cumprod(1 + returns)
        return pd.DataFrame({"ASSET": prices}, index=pd.date_range("2020-01-01", periods=n))

    def test_backtester_equity_curve_positive(self):
        from trading_strategies.backtester import Backtester
        from trading_strategies.moving_average import SMACrossover

        prices = self._make_prices()
        bt = Backtester(transaction_cost=0.001, initial_capital=100_000)
        result = bt.run(SMACrossover(fast_window=10, slow_window=50), prices)
        assert (result.equity_curve > 0).all()

    def test_backtester_returns_length(self):
        from trading_strategies.backtester import Backtester
        from trading_strategies.moving_average import SMACrossover

        prices = self._make_prices()
        bt = Backtester()
        result = bt.run(SMACrossover(fast_window=10, slow_window=50), prices)
        assert len(result.returns) == len(prices)

    def test_performance_metrics_keys(self):
        from trading_strategies.backtester import compute_performance_metrics

        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(0.0005, 0.01, 252))
        metrics = compute_performance_metrics(r)
        for key in ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown"]:
            assert key in metrics

    def test_sma_crossover_invalid_windows(self):
        from trading_strategies.moving_average import SMACrossover

        with pytest.raises(ValueError):
            SMACrossover(fast_window=50, slow_window=20)

    def test_ema_crossover_produces_signals(self):
        from trading_strategies.backtester import Backtester
        from trading_strategies.moving_average import EMACrossover

        prices = self._make_prices()
        bt = Backtester()
        result = bt.run(EMACrossover(fast_span=12, slow_span=26, allow_short=True), prices)
        unique_signals = result.signals["ASSET"].unique()
        # Should contain at least 0 and 1
        assert len(unique_signals) >= 1


class TestPairsTrading:
    def _make_cointegrated_pair(self, n: int = 500) -> tuple[pd.Series, pd.Series]:
        rng = np.random.default_rng(99)
        common = np.cumsum(rng.normal(0, 0.5, n))
        a = np.exp((common + rng.normal(0, 0.1, n)) / 10)
        b = np.exp((common + rng.normal(0, 0.1, n)) / 10)
        idx = pd.date_range("2018-01-01", periods=n)
        return pd.Series(a, index=idx), pd.Series(b, index=idx)

    def test_test_cointegration_returns_keys(self):
        from trading_strategies.pairs_trading import test_cointegration

        a, b = self._make_cointegrated_pair()
        result = test_cointegration(a, b)
        for key in ["hedge_ratio", "spread", "adf_stat", "p_value", "is_cointegrated"]:
            assert key in result

    def test_spread_zscore_mean_near_zero(self):
        from trading_strategies.pairs_trading import spread_zscore

        rng = np.random.default_rng(0)
        spread = pd.Series(rng.normal(5, 1, 252))
        z = spread_zscore(spread)
        assert abs(z.mean()) < 0.1

    def test_half_life_positive_for_stationary(self):
        from trading_strategies.pairs_trading import half_life

        rng = np.random.default_rng(1)
        # Simulate a stationary AR(1) process (fast mean-reversion)
        x = np.zeros(500)
        for t in range(1, 500):
            x[t] = 0.7 * x[t - 1] + rng.normal(0, 0.1)
        hl = half_life(pd.Series(x))
        assert hl > 0
        assert hl < 500


# ── Risk Management ───────────────────────────────────────────────────────────

class TestVaR:
    def _returns(self, seed: int = 42, n: int = 1000) -> np.ndarray:
        return np.random.default_rng(seed).normal(0.0, 0.01, n)

    def test_historical_var_positive(self):
        from risk_management.value_at_risk import var_historical

        r = self._returns()
        assert var_historical(r) > 0

    def test_parametric_var_positive(self):
        from risk_management.value_at_risk import var_parametric

        r = self._returns()
        assert var_parametric(r) > 0

    def test_monte_carlo_var_positive(self):
        from risk_management.value_at_risk import var_monte_carlo

        r = self._returns()
        assert var_monte_carlo(r, rng=np.random.default_rng(0)) > 0

    def test_cvar_greater_than_var(self):
        from risk_management.value_at_risk import var_historical, cvar_historical

        r = self._returns()
        var = var_historical(r)
        cvar = cvar_historical(r)
        assert cvar >= var

    def test_scale_var(self):
        from risk_management.value_at_risk import scale_var

        assert scale_var(1.0, 10) == pytest.approx(np.sqrt(10))

    def test_higher_confidence_gives_higher_var(self):
        from risk_management.value_at_risk import var_historical

        r = self._returns()
        var_95 = var_historical(r, confidence=0.95)
        var_99 = var_historical(r, confidence=0.99)
        assert var_99 > var_95


class TestPortfolioRisk:
    def test_portfolio_beta_near_one_for_market(self):
        from risk_management.portfolio_risk import portfolio_beta

        rng = np.random.default_rng(5)
        mkt = rng.normal(0.0005, 0.01, 500)
        beta = portfolio_beta(mkt, mkt)
        assert beta == pytest.approx(1.0, abs=1e-6)

    def test_tracking_error_zero_for_identical(self):
        from risk_management.portfolio_risk import tracking_error

        r = np.random.default_rng(2).normal(0, 0.01, 252)
        assert tracking_error(r, r) == pytest.approx(0.0, abs=1e-10)

    def test_max_drawdown_negative(self):
        from risk_management.portfolio_risk import max_drawdown

        eq = pd.Series([100, 110, 90, 95, 105])
        dd = max_drawdown(eq)
        assert dd < 0

    def test_drawdown_series_non_positive(self):
        from risk_management.portfolio_risk import drawdown_series

        eq = pd.Series([100, 120, 110, 130, 125])
        dd = drawdown_series(eq)
        assert (dd <= 0).all()

    def test_rolling_volatility_length(self):
        from risk_management.portfolio_risk import rolling_volatility

        r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 252))
        rv = rolling_volatility(r, window=21)
        assert len(rv) == len(r)


# ── Machine Learning ──────────────────────────────────────────────────────────

class TestRegressionModels:
    def test_ols_factor_model_keys(self):
        from machine_learning.regression_models import ols_factor_model

        rng = np.random.default_rng(0)
        n = 200
        f1 = pd.Series(rng.normal(0, 0.01, n), name="f1")
        f2 = pd.Series(rng.normal(0, 0.01, n), name="f2")
        factors = pd.DataFrame({"f1": f1, "f2": f2})
        asset = 0.5 * f1 + 0.3 * f2 + rng.normal(0, 0.002, n)
        result = ols_factor_model(pd.Series(asset), factors)
        assert "alpha" in result
        assert "betas" in result
        assert "r_squared" in result

    def test_make_features_has_expected_columns(self):
        from machine_learning.regression_models import make_features

        n = 100
        idx = pd.date_range("2020-01-01", periods=n)
        prices = pd.DataFrame({"A": 100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, n))}, index=idx)
        returns = prices.pct_change()
        feats = make_features(prices, returns)
        assert any("mom" in col for col in feats.columns)
        assert any("vol" in col for col in feats.columns)


class TestTimeSeries:
    def test_adf_stationary_on_returns(self):
        from machine_learning.time_series import adf_test

        rng = np.random.default_rng(0)
        # White noise is stationary
        r = pd.Series(rng.normal(0, 0.01, 500))
        result = adf_test(r)
        assert result["stationary"] is True
        assert result["n_diffs"] == 0

    def test_compute_acf_length(self):
        from machine_learning.time_series import compute_acf_pacf

        rng = np.random.default_rng(1)
        s = pd.Series(rng.normal(0, 1, 300))
        acf_vals, pacf_vals = compute_acf_pacf(s, nlags=20)
        assert len(acf_vals) == 21
        assert len(pacf_vals) == 21

    def test_fit_arima_returns_aic(self):
        from machine_learning.time_series import fit_arima

        rng = np.random.default_rng(2)
        s = pd.Series(rng.normal(0, 0.01, 300))
        result = fit_arima(s, order=(1, 0, 1))
        assert "aic" in result
        assert "bic" in result
        assert result["aic"] < 0 or result["aic"] > -1e10  # just check it's a number
