import pandas as pd
import numpy as np
import os


class RiskCalculator:
    def __init__(self, data_dir=None):
        if data_dir is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(base_path, '..', 'data', 'raw')
        else:
            self.data_dir = data_dir

        self.stock_prices  = None
        self.market_prices = None
        self.returns       = None
        self.market_returns = None
        self.tickers       = None

        # Cache for per-stock technical metrics (computed once, reused)
        self._stock_technical_cache: dict = {}

    # ──────────────────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────────────────

    def load_data(self):
        """Load locally saved CSV data for stocks and market."""
        stocks_path = os.path.join(self.data_dir, "stocks_prices.csv")
        market_path = os.path.join(self.data_dir, "market_prices.csv")

        self.stock_prices = pd.read_csv(
            stocks_path, index_col=0, parse_dates=True
        )
        self.market_prices = pd.read_csv(
            market_path, index_col=0, parse_dates=True
        )
        self.market_prices.columns = ['TASI_Index']
        self.tickers = self.stock_prices.columns.tolist()

        common_index = self.stock_prices.index.intersection(
            self.market_prices.index
        )
        self.stock_prices  = self.stock_prices.loc[common_index]
        self.market_prices = self.market_prices.loc[common_index]

    # ──────────────────────────────────────────────────────────
    # Return Calculation
    # ──────────────────────────────────────────────────────────

    def calculate_daily_returns(self):
        """
        Logarithmic daily returns: R_t = ln(P_t / P_{t-1})
        Also pre-computes and caches all per-stock technical metrics.
        """
        self.returns = np.log(
            self.stock_prices / self.stock_prices.shift(1)
        ).dropna()

        self.market_returns = np.log(
            self.market_prices / self.market_prices.shift(1)
        ).dropna()

        common_index = self.returns.index.intersection(
            self.market_returns.index
        )
        self.returns        = self.returns.loc[common_index]
        self.market_returns = self.market_returns.loc[common_index]

        # Pre-compute and cache stock-level technical metrics
        self._build_technical_cache()

    # ──────────────────────────────────────────────────────────
    # Per-Stock Technical Metrics Cache
    # ──────────────────────────────────────────────────────────

    def _build_technical_cache(self):
        """
        Compute and store the three new technical risk indicators for
        every ticker.  Called once after calculate_daily_returns().
        Methodology references (Risk_Assessment_MathML):
          §3.2 — Downside Volatility, Maximum Drawdown, Amihud Illiquidity
        """
        self._stock_technical_cache = {}

        for ticker in self.tickers:
            if ticker not in self.returns.columns:
                continue
            r = self.returns[ticker]

            # ── A. Downside Volatility ─────────────────────────────
            # σ_down = std of min(R_t, 0) series * sqrt(252)
            # Only negative returns contribute; zero counts as zero.
            neg_returns     = r.apply(lambda x: min(x, 0.0))
            mean_neg        = neg_returns.mean()
            downside_var    = ((neg_returns - mean_neg) ** 2).mean()
            downside_vol    = np.sqrt(downside_var) * np.sqrt(252)   # annualised

            # ── B. Maximum Drawdown ────────────────────────────────
            # MDD = max((peak - trough) / peak) over the full window
            prices      = self.stock_prices[ticker].dropna()
            rolling_max = prices.cummax()
            drawdown    = (rolling_max - prices) / rolling_max
            mdd         = drawdown.max()                              # 0–1 ratio

            # ── C. Amihud Illiquidity ──────────────────────────────
            # ILLIQ = mean(|R_t| / Volume_t)
            # Proxy: we don't store raw volume in the price CSV, so we
            # use the price itself as a proxy denominator (scaled by
            # shares-outstanding proxy = 1).  When volume data is
            # available this should be replaced with actual TV_{i,τ}.
            # Here we compute the Amihud ratio using absolute return
            # divided by normalised price level (a valid approximation
            # for relative illiquidity ranking across assets).
            price_level = self.stock_prices[ticker].dropna()
            aligned_r   = r.loc[r.index.isin(price_level.index)]
            aligned_p   = price_level.loc[aligned_r.index]
            # Avoid division by zero
            valid_mask  = aligned_p > 0
            if valid_mask.sum() > 0:
                amihud = (aligned_r[valid_mask].abs() / aligned_p[valid_mask]).mean()
            else:
                amihud = 0.0

            self._stock_technical_cache[ticker] = {
                'Downside_Volatility': float(downside_vol),
                'Max_Drawdown'       : float(mdd),
                'Amihud_Illiquidity' : float(amihud),
            }

    # ──────────────────────────────────────────────────────────
    # Public API for app.py live inference
    # ──────────────────────────────────────────────────────────

    def get_stock_metric(self, ticker: str, metric_name: str):
        """
        Return a single cached technical metric for a given ticker.
        metric_name must be one of:
            'Downside_Volatility', 'Max_Drawdown', 'Amihud_Illiquidity'
        Returns None if ticker or metric is not available.
        """
        entry = self._stock_technical_cache.get(ticker)
        if entry is None:
            return None
        return entry.get(metric_name)

    def get_portfolio_technical_metrics(self, tickers: list, weights: list) -> dict:
        """
        Compute portfolio-weighted averages of the three technical
        indicators for a given ticker/weight combination.
        Returns a dict with keys:
            Portfolio_Downside_Volatility
            Portfolio_Max_Drawdown
            Portfolio_Amihud_Illiquidity
        Missing tickers are skipped; their weight is redistributed.
        """
        result     = {'Portfolio_Downside_Volatility': 0.0,
                      'Portfolio_Max_Drawdown'       : 0.0,
                      'Portfolio_Amihud_Illiquidity' : 0.0}
        total_w    = 0.0

        for t, w in zip(tickers, weights):
            entry = self._stock_technical_cache.get(t)
            if entry is None:
                continue
            result['Portfolio_Downside_Volatility'] += w * entry['Downside_Volatility']
            result['Portfolio_Max_Drawdown']        += w * entry['Max_Drawdown']
            result['Portfolio_Amihud_Illiquidity']  += w * entry['Amihud_Illiquidity']
            total_w += w

        if total_w > 0 and total_w < 1.0:
            # Rescale to handle skipped tickers
            for k in result:
                result[k] /= total_w

        return result

    # ──────────────────────────────────────────────────────────
    # Original Methods (unchanged)
    # ──────────────────────────────────────────────────────────

    def get_individual_metrics(self):
        """
        Individual stock volatility and beta.
        Volatility: sigma_i = std(R_i) * sqrt(252)
        Beta:       beta_i  = Cov(R_i, R_m) / Var(R_m)
        """
        volatility = self.returns.std() * np.sqrt(252)
        market_var = self.market_returns['TASI_Index'].var()

        betas = {}
        for ticker in self.tickers:
            cov_matrix   = np.cov(self.returns[ticker],
                                  self.market_returns['TASI_Index'])
            betas[ticker] = cov_matrix[0, 1] / market_var

        return volatility, pd.Series(betas)

    def calculate_portfolio_risk(self, weights):
        """
        Portfolio Volatility = sqrt(w^T * Sigma * w)
        Portfolio Beta       = sum(w_i * beta_i)
        """
        weights = np.array(weights)
        if len(weights) != len(self.tickers):
            raise ValueError(
                f"Received {len(weights)} weights, "
                f"expected {len(self.tickers)}."
            )

        cov_matrix_annual  = self.returns.cov() * 252
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        individual_vols, individual_betas = self.get_individual_metrics()
        portfolio_beta = np.sum(weights * individual_betas.values)

        return {
            "Portfolio_Volatility_Percentage": round(portfolio_volatility * 100, 2),
            "Portfolio_Beta"                 : round(portfolio_beta, 3),
            "Stock_Betas"                    : individual_betas.to_dict(),
            "Stock_Volatilities"             : individual_vols.to_dict()
        }

    def calculate_sector_metrics(self, sector_tickers):
        """
        Sector volatility and beta from equal-weighted sector portfolio.
        sigma_sector = std(R_sector) * sqrt(252)
        beta_sector  = Cov(R_sector, R_market) / Var(R_market)
        """
        valid = [t for t in sector_tickers if t in self.returns.columns]
        if not valid:
            return 0.15, 1.0

        sector_returns = self.returns[valid].mean(axis=1)
        sector_vol     = sector_returns.std() * np.sqrt(252)

        market_var  = self.market_returns['TASI_Index'].var()
        cov         = np.cov(sector_returns, self.market_returns['TASI_Index'])
        sector_beta = cov[0, 1] / market_var

        return sector_vol, sector_beta
