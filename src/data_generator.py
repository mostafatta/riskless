import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from calculations import RiskCalculator
from risk_labeler  import RiskLabeler
from data_loader   import TadawulDataLoader


def generate_dataset(num_samples=500):
    print(f"--- GENERATING {num_samples} RANDOM PORTFOLIOS ---\n")

    # ──────────────────────────────────────────────────────────
    # 1. Fetch market data & fundamentals
    # ──────────────────────────────────────────────────────────
    loader = TadawulDataLoader()
    loader.fetch_stock_data()
    loader.fetch_market_data()
    loader.fetch_metadata()          # now writes all 8 fundamental columns

    calc = RiskCalculator()
    calc.load_data()
    calc.calculate_daily_returns()   # also builds _stock_technical_cache

    labeler = RiskLabeler()

    # ──────────────────────────────────────────────────────────
    # 2. Load metadata (Market Cap + Sector + new fundamentals)
    # ──────────────────────────────────────────────────────────
    meta_path = os.path.join(loader.data_dir, "stocks_metadata.csv")
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path).set_index("Ticker")
    else:
        meta_df = None
        print("  [WARN] stocks_metadata.csv not found — fundamental features will be NaN")

    available_tickers = calc.tickers

    # ── Helper: pull a fundamental for a single ticker ────────
    def get_fundamental(ticker, col, default=np.nan):
        if meta_df is not None and ticker in meta_df.index and col in meta_df.columns:
            val = meta_df.loc[ticker, col]
            return float(val) if not pd.isna(val) else default
        return default

    dataset = []

    for i in range(num_samples):
        try:
            # ── Random portfolio construction ─────────────────
            num_stocks       = np.random.randint(1, 8)
            selected_tickers = np.random.choice(available_tickers, num_stocks, replace=False)
            weights_raw      = np.random.dirichlet(np.ones(num_stocks), size=1)[0]

            full_weights       = [0.0] * len(available_tickers)
            port_cap_score     = 0.0
            portfolio_sectors  = {}

            # ── Financial fundamentals (portfolio-weighted) ───
            port_de    = 0.0   # Debt-to-Equity
            port_cr    = 0.0   # Current Ratio
            port_icr   = 0.0   # Interest Coverage
            port_roa   = 0.0   # Return on Assets
            port_rgvol = 0.0   # Revenue Growth Volatility

            for idx, ticker in enumerate(selected_tickers):
                w        = weights_raw[idx]
                full_idx = available_tickers.index(ticker)
                full_weights[full_idx] = w

                # Market cap & sector
                if meta_df is not None and ticker in meta_df.index:
                    score  = meta_df.loc[ticker, "Market_Cap_Score"]
                    sector = meta_df.loc[ticker, "Sector"]
                else:
                    score  = 2.0
                    sector = loader.sector_map.get(ticker, "Unknown")

                port_cap_score += w * float(score)
                portfolio_sectors[sector] = portfolio_sectors.get(sector, 0.0) + w

                # Fundamental aggregation (§4 methodology)
                port_de    += w * get_fundamental(ticker, "Debt_to_Equity",     0.5)
                port_cr    += w * get_fundamental(ticker, "Current_Ratio",      1.5)
                port_icr   += w * get_fundamental(ticker, "Interest_Coverage",  5.0)
                port_roa   += w * get_fundamental(ticker, "ROA",                0.05)
                port_rgvol += w * get_fundamental(ticker, "Revenue_Growth_Vol", 0.05)

            # ── Diversification index ─────────────────────────
            div_index = 1.0 - np.sum(np.array(full_weights) ** 2)

            # ── Portfolio volatility & beta ───────────────────
            metrics = calc.calculate_portfolio_risk(full_weights)

            # ── Sector-level volatility & beta ────────────────
            weighted_sector_vol  = 0.0
            weighted_sector_beta = 0.0

            for sec, sec_weight in portfolio_sectors.items():
                sec_tickers = [tk for tk, s in loader.sector_map.items() if s == sec]
                s_vol, s_beta         = calc.calculate_sector_metrics(sec_tickers)
                weighted_sector_vol  += sec_weight * s_vol
                weighted_sector_beta += sec_weight * s_beta

            # ── Technical indicators (from cache) ─────────────
            tech = calc.get_portfolio_technical_metrics(
                list(selected_tickers), list(weights_raw)
            )
            port_downside_vol = tech['Portfolio_Downside_Volatility']
            port_mdd          = tech['Portfolio_Max_Drawdown']
            port_amihud       = tech['Portfolio_Amihud_Illiquidity']

            # ══════════════════════════════════════════════════
            # Dataset Balancing & Injection (original logic)
            # ══════════════════════════════════════════════════
            vol_pct   = metrics['Portfolio_Volatility_Percentage']
            beta      = metrics['Portfolio_Beta']
            rand_val  = np.random.rand()

            if rand_val < 0.35:
                # ── Inject LOW RISK (35%) ─────────────────────
                vol_pct              = np.random.uniform(5.0,  12.0)
                beta                 = np.random.uniform(0.5,   0.8)
                weighted_sector_vol  = np.random.uniform(0.05,  0.12)
                weighted_sector_beta = np.random.uniform(0.5,   0.8)
                port_cap_score       = np.random.uniform(2.5,   3.0)
                div_index            = np.random.uniform(0.7,   0.9)
                # Align new technical/financial features to low-risk profile
                port_downside_vol    = np.random.uniform(0.03,  0.08)
                port_mdd             = np.random.uniform(0.02,  0.08)
                port_amihud          = np.random.uniform(1e-6,  5e-6)
                port_de              = np.random.uniform(0.1,   0.5)
                port_cr              = np.random.uniform(2.0,   4.0)
                port_icr             = np.random.uniform(8.0,  20.0)
                port_roa             = np.random.uniform(0.08,  0.20)
                port_rgvol           = np.random.uniform(0.01,  0.04)

            elif rand_val > 0.70:
                # ── Inject HIGH RISK (30%) ────────────────────
                vol_pct              = np.random.uniform(28.0, 40.0)
                beta                 = np.random.uniform(1.2,   1.8)
                weighted_sector_vol  = np.random.uniform(0.28,  0.40)
                weighted_sector_beta = np.random.uniform(1.2,   1.6)
                port_cap_score       = np.random.uniform(1.0,   1.5)
                div_index            = np.random.uniform(0.0,   0.3)
                # Align new features to high-risk profile
                port_downside_vol    = np.random.uniform(0.18,  0.35)
                port_mdd             = np.random.uniform(0.25,  0.55)
                port_amihud          = np.random.uniform(2e-5,  8e-5)
                port_de              = np.random.uniform(1.5,   4.0)
                port_cr              = np.random.uniform(0.5,   1.1)
                port_icr             = np.random.uniform(0.5,   2.5)
                port_roa             = np.random.uniform(-0.10, 0.02)
                port_rgvol           = np.random.uniform(0.15,  0.40)

            # Remaining 35% → natural MEDIUM RISK values (computed above)

            # ── Math label ────────────────────────────────────
            result = labeler.calculate_final_score(
                port_q_pct=vol_pct,
                port_b=beta,
                sector_q=weighted_sector_vol,
                sector_b=weighted_sector_beta
            )

            dataset.append({
                # ── Original 6 features ──────────────────────
                "Portfolio_Volatility"        : round(vol_pct,   2),
                "Portfolio_Beta"              : round(beta,       3),
                "Sector_Volatility"           : round(weighted_sector_vol  * 100, 2),
                "Sector_Beta"                 : round(weighted_sector_beta, 2),
                "Diversification_Index"       : round(div_index,  3),
                "Market_Cap_Score"            : round(port_cap_score, 2),
                # ── New 8 features ───────────────────────────
                "Portfolio_Downside_Volatility": round(port_downside_vol * 100, 4),
                "Portfolio_Max_Drawdown"       : round(port_mdd,           4),
                "Portfolio_Amihud_Illiquidity" : round(port_amihud,        8),
                "Portfolio_Debt_to_Equity"     : round(port_de,            4),
                "Portfolio_Current_Ratio"      : round(port_cr,            4),
                "Portfolio_Interest_Coverage"  : round(port_icr,           4),
                "Portfolio_ROA"                : round(port_roa,           4),
                "Portfolio_Revenue_Growth_Vol" : round(port_rgvol,         4),
                # ── Labels ───────────────────────────────────
                "Risk_Score"                   : result['Final_Risk_Score'],
                "Risk_Category"                : result['Risk_Category'],
            })

        except Exception as e:
            print(f"  [WARN] Sample {i} skipped: {e}")
            continue

    df = pd.DataFrame(dataset)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data', 'processed', 'portfolio_dataset.csv'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*55}")
    print(f"  Generated {len(df)} portfolios  |  {len(df.columns)} columns")
    print(f"{'='*55}")
    print("\n🔥 Risk Category Distribution:")
    print(df['Risk_Category'].value_counts())
    print(f"\nColumns: {df.columns.tolist()}")
    return df


if __name__ == "__main__":
    generate_dataset(500)
