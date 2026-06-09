import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


class TadawulDataLoader:
    def __init__(self, tickers=None, data_dir=None):
        if tickers is None:
            self.tickers = [
                # === Banks (القطاع البنكي) ===
                '1120.SR', '1010.SR', '1180.SR', '1080.SR',
                # === Energy & Industry (الطاقة والصناعة) ===
                '2222.SR', '2010.SR', '2310.SR', '2020.SR',
                # === Telecom & Retail (الاتصالات والتجزئة) ===
                '7010.SR', '4190.SR', '4200.SR', '4030.SR',
                # === Cement (الاسمنت) ===
                '3030.SR', '3040.SR', '3050.SR', '3060.SR',
                # === Services & Mining (خدمات وتعدين) ===
                '4003.SR', '4008.SR', '2150.SR', '1211.SR'
            ]
        else:
            self.tickers = tickers

        self.sector_map = {
            '1120.SR': 'Banks',    '1010.SR': 'Banks',
            '1180.SR': 'Banks',    '1080.SR': 'Banks',
            '2222.SR': 'Energy',   '2010.SR': 'Energy',
            '2310.SR': 'Industry', '2020.SR': 'Industry',
            '7010.SR': 'Telecom',  '4190.SR': 'Retail',
            '4200.SR': 'Retail',   '4030.SR': 'Retail',
            '3030.SR': 'Cement',   '3040.SR': 'Cement',
            '3050.SR': 'Cement',   '3060.SR': 'Cement',
            '4003.SR': 'Services', '4008.SR': 'Services',
            '2150.SR': 'Mining',   '1211.SR': 'Mining'
        }

        self.market_ticker = "^TASI.SR"

        if data_dir is None:
            base = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(base, '..', 'data', 'raw')
        else:
            self.data_dir = data_dir

        today        = datetime.today()
        one_year_ago = today - timedelta(days=365)
        self.end_date   = today.strftime('%Y-%m-%d')
        self.start_date = one_year_ago.strftime('%Y-%m-%d')

        # Extended window for revenue growth volatility (needs multiple periods)
        two_years_ago        = today - timedelta(days=730)
        self.start_date_long = two_years_ago.strftime('%Y-%m-%d')

        os.makedirs(self.data_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Original methods (unchanged)
    # ──────────────────────────────────────────────────────────

    def fetch_stock_data(self):
        """Download stock prices from Yahoo Finance."""
        print(f"Fetching data for {len(self.tickers)} stocks...")
        try:
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False
            )['Adj Close']

            if isinstance(data, pd.Series):
                data = data.to_frame()

            file_path = os.path.join(self.data_dir, "stocks_prices.csv")
            data.to_csv(file_path)
            print(f"  Stock prices saved to {file_path}")
            return data
        except Exception as e:
            print(f"  Error downloading stocks: {e}")
            return None

    def fetch_market_data(self):
        """Download TASI market index from Yahoo Finance."""
        print(f"Fetching Market Index ({self.market_ticker})...")
        try:
            market_data = yf.download(
                self.market_ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False
            )['Adj Close']

            market_data.name = "TASI_Index"
            file_path = os.path.join(self.data_dir, "market_prices.csv")
            market_data.to_csv(file_path)
            print(f"  Market data saved to {file_path}")
            return market_data
        except Exception as e:
            print(f"  Error downloading market data: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # Updated fetch_metadata — adds 5 new fundamental columns
    # ──────────────────────────────────────────────────────────

    def fetch_metadata(self):
        """
        Fetch per-stock fundamental data from Yahoo Finance.

        Original columns (unchanged):
            Ticker, Market_Cap_Score, Sector

        New columns added (methodology §3.3 — Financial Indicators):
            Debt_to_Equity      — Total Debt / Equity  (positively related to risk)
            Current_Ratio       — Current Assets / Current Liabilities (negatively related)
            Interest_Coverage   — EBIT / Interest Expense  (negatively related)
            ROA                 — Net Income / Total Assets (negatively related)
            Revenue_Growth_Vol  — Std of quarterly revenue growth rates (positively related)

        All fundamentals are pulled from yfinance .info and .financials.
        Robust fallbacks are applied so a single bad ticker never breaks the run.
        """
        print("Fetching metadata + fundamentals...")
        metadata_list = []

        for t in self.tickers:
            row = {
                "Ticker"            : t,
                "Sector"            : self.sector_map.get(t, "Unknown"),
                # ── Original ──────────────────────────────
                "Market_Cap_Score"  : 2.0,   # default Mid Cap
                # ── New fundamentals ──────────────────────
                "Debt_to_Equity"    : np.nan,
                "Current_Ratio"     : np.nan,
                "Interest_Coverage" : np.nan,
                "ROA"               : np.nan,
                "Revenue_Growth_Vol": np.nan,
            }

            try:
                stock = yf.Ticker(t)
                info  = stock.info

                # ── Market Cap Score (original logic) ─────
                mkt_cap = info.get('marketCap', 0) or 0
                if mkt_cap > 50_000_000_000:
                    row["Market_Cap_Score"] = 3.0   # Large Cap
                elif mkt_cap > 10_000_000_000:
                    row["Market_Cap_Score"] = 2.0   # Mid Cap
                else:
                    row["Market_Cap_Score"] = 1.0   # Small Cap

                # ── Debt-to-Equity ─────────────────────────
                # yfinance provides debtToEquity directly (already ratio×100 in some versions)
                de = info.get('debtToEquity')
                if de is not None:
                    # yfinance returns D/E as percentage in some versions; normalise to ratio
                    row["Debt_to_Equity"] = float(de) / 100.0 if float(de) > 20 else float(de)

                # ── Current Ratio ──────────────────────────
                cr = info.get('currentRatio')
                if cr is not None:
                    row["Current_Ratio"] = float(cr)

                # ── Return on Assets ───────────────────────
                roa = info.get('returnOnAssets')
                if roa is not None:
                    row["ROA"] = float(roa)

                # ── Interest Coverage & Revenue Growth Vol ─
                # These need income statement data
                try:
                    financials = stock.financials   # annual, most-recent columns first

                    # Interest Coverage = EBIT / Interest Expense
                    if financials is not None and not financials.empty:
                        ebit_row  = _find_row(financials, ['EBIT', 'Operating Income'])
                        int_row   = _find_row(financials, ['Interest Expense',
                                                            'Interest And Debt Expense'])

                        if ebit_row is not None and int_row is not None:
                            ebit_val = _first_valid(financials.loc[ebit_row])
                            int_val  = _first_valid(financials.loc[int_row])
                            if int_val and int_val != 0:
                                row["Interest_Coverage"] = float(ebit_val) / abs(float(int_val))

                    # Revenue Growth Volatility
                    # σ(RG) = std of quarterly revenue growth rates
                    try:
                        qfinancials = stock.quarterly_financials
                        if qfinancials is not None and not qfinancials.empty:
                            rev_row = _find_row(qfinancials, ['Total Revenue', 'Revenue'])
                            if rev_row is not None:
                                rev_series = qfinancials.loc[rev_row].dropna().sort_index()
                                if len(rev_series) >= 3:
                                    rev_vals   = rev_series.values.astype(float)
                                    # growth rate: (R_t - R_{t-1}) / |R_{t-1}|
                                    growth     = np.diff(rev_vals) / np.abs(rev_vals[:-1] + 1e-9)
                                    row["Revenue_Growth_Vol"] = float(np.std(growth))
                    except Exception:
                        pass

                except Exception:
                    pass

            except Exception as e:
                print(f"  [WARN] Could not fetch fundamentals for {t}: {e}")

            metadata_list.append(row)
            print(f"  {t} → Cap={row['Market_Cap_Score']} | D/E={row['Debt_to_Equity']:.3f} "
                  f"| CR={row['Current_Ratio']} | ICR={row['Interest_Coverage']} "
                  f"| ROA={row['ROA']} | RevGVol={row['Revenue_Growth_Vol']}"
                  if not any(pd.isna(v) for v in [row['Debt_to_Equity'], row['Current_Ratio'],
                                                   row['Interest_Coverage'], row['ROA'],
                                                   row['Revenue_Growth_Vol']])
                  else f"  {t} → Cap={row['Market_Cap_Score']} (some fundamentals N/A)")

        df = pd.DataFrame(metadata_list)

        # ── Column-level median imputation for any NaN fundamentals ──
        for col in ["Debt_to_Equity", "Current_Ratio", "Interest_Coverage",
                    "ROA", "Revenue_Growth_Vol"]:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                median_val = df[col].median()
                # If all are NaN use sensible sector-neutral defaults
                if pd.isna(median_val):
                    defaults = {
                        "Debt_to_Equity"    : 0.5,
                        "Current_Ratio"     : 1.5,
                        "Interest_Coverage" : 5.0,
                        "ROA"               : 0.05,
                        "Revenue_Growth_Vol": 0.05,
                    }
                    median_val = defaults[col]
                df[col] = df[col].fillna(median_val)
                print(f"  [IMPUTE] {col}: filled {n_nan} NaN(s) with median={median_val:.4f}")

        file_path = os.path.join(self.data_dir, "stocks_metadata.csv")
        df.to_csv(file_path, index=False)
        print(f"  Metadata saved to {file_path}  ({len(df)} tickers, {len(df.columns)} columns)")
        return df


# ──────────────────────────────────────────────────────────────
# Private helpers for navigating yfinance financials DataFrames
# ──────────────────────────────────────────────────────────────

def _find_row(df: pd.DataFrame, candidates: list):
    """Return the first index label that matches any candidate string (case-insensitive)."""
    for label in df.index:
        for c in candidates:
            if c.lower() in str(label).lower():
                return label
    return None


def _first_valid(series: pd.Series):
    """Return the first non-NaN value from a pandas Series."""
    for v in series:
        if v is not None and not pd.isna(v):
            return v
    return None
