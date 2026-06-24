import sys
import os

# --- Fix Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, 'src')
sys.path.append(SRC_DIR)
sys.path.append(BASE_DIR)

import joblib
import pandas as pd
import numpy as np

from src.data_loader  import TadawulDataLoader
from src.calculations import RiskCalculator
from src.risk_labeler import RiskLabeler


# ══════════════════════════════════════════════
#  MODEL REGISTRY
#  كل مودل له: اسم، مسار الملف، ونوعه
# ══════════════════════════════════════════════
MODELS = {
    "1": {
        "name"    : "Random Forest  (Rolling Window)",
        "type"    : "rf",
        "model"   : "rf_rolling_window.pkl",
        "scaler"  : None,
        "encoder" : None,
        "features": "rf_features.pkl",
    },
    "2": {
        "name"    : "LSTM           (Rolling Window)",
        "type"    : "lstm",
        "model"   : "lstm_rolling_window.keras",
        "scaler"  : "lstm_scaler.pkl",
        "encoder" : "lstm_label_encoder.pkl",
        "features": "lstm_features.pkl",
    },
    "3": {
        "name"    : "SVM            (Rolling Window)",
        "type"    : "svm",
        "model"   : "svm_rolling_window.pkl",
        "scaler"  : "svm_scaler.pkl",
        "encoder" : "svm_label_encoder.pkl",
        "features": "svm_features.pkl",
    },
}


# ══════════════════════════════════════════════
#  MODEL SELECTION
# ══════════════════════════════════════════════
def select_model():
    """اختيار المودل من القائمة في بداية البرنامج."""
    print("\n" + "═" * 45)
    print("       SELECT PREDICTION MODEL")
    print("═" * 45)
    for key, info in MODELS.items():
        print(f"  [{key}]  {info['name']}")
    print("═" * 45)

    while True:
        choice = input("  Enter choice (1 / 2 / 3): ").strip()
        if choice in MODELS:
            selected = MODELS[choice]
            print(f"\n  ✓ Selected: {selected['name']}\n")
            return selected
        print("  Invalid choice. Please enter 1, 2, or 3.")


# ══════════════════════════════════════════════
#  LOAD MODEL ARTIFACTS
# ══════════════════════════════════════════════
def load_model_artifacts(model_info):
    """
    يحمّل الملفات المطلوبة لكل مودل:
    - RF  : .pkl فقط
    - LSTM: .keras + scaler + label_encoder
    - SVM : .pkl  + scaler + label_encoder
    """
    models_dir = os.path.join(BASE_DIR, "models")

    model_path = os.path.join(models_dir, model_info["model"])
    if not os.path.exists(model_path):
        print(f"\n  [Error] Model file not found: {model_path}")
        print("  Run the corresponding training script first.\n")
        return None, None, None

    mtype = model_info["type"]

    if mtype == "lstm":
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
    else:
        model = joblib.load(model_path)

    scaler  = None
    encoder = None

    if model_info["scaler"]:
        scaler_path = os.path.join(models_dir, model_info["scaler"])
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            print(f"  [Warning] Scaler not found: {scaler_path}")

    if model_info["encoder"]:
        encoder_path = os.path.join(models_dir, model_info["encoder"])
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
        else:
            print(f"  [Warning] LabelEncoder not found: {encoder_path}")

    features = []
    if "features" in model_info and model_info["features"]:
        feat_path = os.path.join(models_dir, model_info["features"])
        if os.path.exists(feat_path):
            features = joblib.load(feat_path)

    return model, scaler, encoder, features


# ══════════════════════════════════════════════
#  PREDICTION LOGIC
# ══════════════════════════════════════════════
def predict_with_model(model, scaler, encoder, model_info, feature_values, feature_cols):
    """
    يُنفّذ التنبؤ بناءً على نوع المودل ويرجع:
    (ai_category, prob_str)
    """
    mtype = model_info["type"]

    input_df = pd.DataFrame([feature_values], columns=feature_cols)
    X_raw    = input_df.values

    # ── Random Forest ──────────────────────────
    if mtype == "rf":
        ai_category = model.predict(input_df)[0]
        probs       = model.predict_proba(input_df)[0]
        classes     = model.classes_
        prob_str    = " | ".join(
            [f"{c}: {p*100:.0f}%" for c, p in zip(classes, probs)]
        )

    # ── LSTM ───────────────────────────────────
    elif mtype == "lstm":
        import numpy as np
        X_scaled = scaler.transform(X_raw) if scaler else X_raw
        X_3d     = X_scaled.reshape(1, 1, X_scaled.shape[1])
        probs    = model.predict(X_3d, verbose=0)[0]

        pred_idx    = int(np.argmax(probs))
        classes     = encoder.classes_ if encoder else [str(i) for i in range(len(probs))]
        ai_category = classes[pred_idx]
        prob_str    = " | ".join(
            [f"{c}: {p*100:.0f}%" for c, p in zip(classes, probs)]
        )

    # ── SVM ────────────────────────────────────
    elif mtype == "svm":
        X_scaled    = scaler.transform(X_raw) if scaler else X_raw
        pred_encoded = model.predict(X_scaled)[0]
        ai_category  = encoder.inverse_transform([pred_encoded])[0] if encoder else str(pred_encoded)

        if hasattr(model, "predict_proba"):
            probs    = model.predict_proba(X_scaled)[0]
            classes  = encoder.classes_ if encoder else model.classes_
            prob_str = " | ".join(
                [f"{c}: {p*100:.0f}%" for c, p in zip(classes, probs)]
            )
        else:
            prob_str = "N/A (probability=False)"

    else:
        ai_category = "Unknown model type"
        prob_str    = "N/A"

    return ai_category, prob_str


# ══════════════════════════════════════════════
#  PORTFOLIO INPUT
# ══════════════════════════════════════════════
def get_user_portfolio():
    """Get portfolio tickers and weights from user input."""
    tickers, weights = [], []
    print("\n" + "═" * 45)
    print("         NEW PORTFOLIO ENTRY")
    print("═" * 45)

    try:
        num_stocks = int(input(
            "How many stocks in your portfolio? (e.g. 3): "
        ))
        if num_stocks <= 0:
            print("Error: Must have at least 1 stock.")
            return None, None
    except ValueError:
        print("Error: Please enter a valid number.")
        return None, None

    remaining_weight = 100.0
    for i in range(num_stocks):
        ticker = input(f"\n  Stock #{i+1} Ticker (e.g., 2222.SR): ").strip()
        if ticker and not ticker.upper().endswith('.SR'):
            ticker += ".SR"

        raw_w  = input(f"  Weight % (Remaining: {remaining_weight:.1f}%): ").replace('%','').strip()
        w_val  = float(raw_w) if raw_w else 0.0

        tickers.append(ticker.upper())
        weights.append(w_val / 100.0)
        remaining_weight -= w_val

    print(f"\n  Portfolio Summary:")
    print(f"  {'Ticker':<15}{'Weight':<10}")
    print(f"  {'-'*22}")
    for t, w in zip(tickers, weights):
        print(f"  {t:<15}{w*100:.1f}%")

    return tickers, weights


# ══════════════════════════════════════════════
#  MAIN PROCESS
# ══════════════════════════════════════════════
def process_prediction(tickers, weights, model, scaler, encoder, features_to_use, model_info):
    """Run calculations and AI prediction, then display results."""
    print(f"\n{'─'*45}")
    print("  PROCESSING... PLEASE WAIT")
    print(f"{'─'*45}")

    try:
        data_directory = os.path.join(BASE_DIR, 'data', 'raw')

        # ── Fetch Data ─────────────────────────
        loader = TadawulDataLoader(tickers=tickers, data_dir=data_directory)
        loader.fetch_stock_data()
        loader.fetch_market_data()

        meta_path = os.path.join(loader.data_dir, "stocks_metadata.csv")
        if not os.path.exists(meta_path):
            loader.fetch_metadata()
        meta_df = pd.read_csv(meta_path).set_index("Ticker")

        # ── Calculations ───────────────────────
        calc = RiskCalculator(data_dir=data_directory)
        calc.load_data()
        calc.calculate_daily_returns()

        metrics = calc.calculate_portfolio_risk(weights)
        vol     = metrics['Portfolio_Volatility_Percentage']
        beta    = metrics['Portfolio_Beta']

        div_index = 1.0 - np.sum(np.array(weights) ** 2)

        portfolio_sectors = {}
        port_cap_score    = 0.0
        for t, w in zip(tickers, weights):
            score  = meta_df.loc[t, "Market_Cap_Score"] if t in meta_df.index else 2.0
            port_cap_score += w * score
            sector = (
                meta_df.loc[t, "Sector"]
                if (t in meta_df.index and "Sector" in meta_df.columns)
                else loader.sector_map.get(t, "Unknown")
            )
            portfolio_sectors[sector] = portfolio_sectors.get(sector, 0.0) + w

        weighted_sector_vol  = 0.0
        weighted_sector_beta = 0.0
        for sec, sec_weight in portfolio_sectors.items():
            sec_tickers = [tk for tk, s in loader.sector_map.items() if s == sec]
            s_vol, s_beta = calc.calculate_sector_metrics(sec_tickers)
            weighted_sector_vol  += sec_weight * s_vol
            weighted_sector_beta += sec_weight * s_beta

        # ── Math Score ─────────────────────────
        labeler      = RiskLabeler()
        score_result = labeler.calculate_final_score(
            port_q_pct=vol,
            port_b=beta,
            sector_q=weighted_sector_vol,
            sector_b=weighted_sector_beta
        )

        # ── AI Prediction ──────────────────────
        technical_metrics = calc.get_portfolio_technical_metrics(tickers, weights)
        port_downside_vol = technical_metrics.get('Portfolio_Downside_Volatility', 0.5)
        port_max_drawdown = technical_metrics.get('Portfolio_Max_Drawdown', 0.5)
        port_amihud = technical_metrics.get('Portfolio_Amihud_Illiquidity', 0.5)

        port_debt_to_equity = 0.0
        port_revenue_growth_vol = 0.0
        port_current_ratio = 0.0
        port_interest_coverage = 0.0
        port_roa = 0.0
        total_w = sum(weights)

        for t, w in zip(tickers, weights):
            if t in meta_df.index:
                def get_val(col, default=0.5):
                    v = meta_df.loc[t, col]
                    try:
                        v_float = float(v)
                        return v_float if not np.isnan(v_float) else default
                    except (Exception):
                        return default

                port_debt_to_equity += w * get_val("Debt_to_Equity", 0.5)
                port_revenue_growth_vol += w * get_val("Revenue_Growth_Vol", 0.05)
                port_current_ratio += w * get_val("Current_Ratio", 1.5)
                port_interest_coverage += w * get_val("Interest_Coverage", 5.0)
                port_roa += w * get_val("ROA", 0.05)
            else:
                port_debt_to_equity += w * 0.5
                port_revenue_growth_vol += w * 0.05
                port_current_ratio += w * 1.5
                port_interest_coverage += w * 5.0
                port_roa += w * 0.05

        if total_w > 0:
            port_debt_to_equity /= total_w
            port_revenue_growth_vol /= total_w
            port_current_ratio /= total_w
            port_interest_coverage /= total_w
            port_roa /= total_w

        base_values = {
            'Portfolio_Volatility': vol,
            'Portfolio_Beta': beta,
            'Sector_Volatility': weighted_sector_vol * 100,
            'Sector_Beta': weighted_sector_beta,
            'Diversification_Index': div_index,
            'Market_Cap_Score': port_cap_score,
            'Portfolio_Downside_Volatility': port_downside_vol,
            'Portfolio_Max_Drawdown': port_max_drawdown,
            'Portfolio_Amihud_Illiquidity': port_amihud,
            'Portfolio_Debt_to_Equity': port_debt_to_equity,
            'Portfolio_Revenue_Growth_Vol': port_revenue_growth_vol,
            'Portfolio_Current_Ratio': port_current_ratio,
            'Portfolio_Interest_Coverage': port_interest_coverage,
            'Portfolio_ROA': port_roa,
        }

        ALL_FEATURES = [
            'Portfolio_Volatility', 'Portfolio_Beta',
            'Sector_Volatility', 'Sector_Beta',
            'Diversification_Index', 'Market_Cap_Score',
            'Portfolio_Downside_Volatility', 'Portfolio_Max_Drawdown',
            'Portfolio_Amihud_Illiquidity', 'Portfolio_Debt_to_Equity',
            'Portfolio_Revenue_Growth_Vol', 'Portfolio_Current_Ratio',
            'Portfolio_Interest_Coverage', 'Portfolio_ROA'
        ]
        feat_list = features_to_use if features_to_use and len(features_to_use) > 0 else ALL_FEATURES
        feature_values = [base_values.get(f, 0.5) for f in feat_list]

        ai_category, prob_str = predict_with_model(
            model, scaler, encoder, model_info, feature_values, feat_list
        )

        # ── Display Results ────────────────────
        print("\n" + "★" * 45)
        print("          RISK ANALYSIS RESULTS")
        print(f"          Model: {model_info['name'].strip()}")
        print("★" * 45)

        print(f"\n  {'METRIC':<30}{'VALUE':>12}")
        print(f"  {'─'*42}")
        print(f"  {'Portfolio Volatility':<30}{vol:>11.2f}%")
        print(f"  {'Portfolio Beta':<30}{beta:>12.3f}")
        print(f"  {'Diversification Index':<30}{div_index:>12.3f}")
        print(f"  {'Market Cap Score':<30}{port_cap_score:>12.2f}")
        print(f"  {'Sector Volatility':<30}{weighted_sector_vol*100:>11.2f}%")
        print(f"  {'Sector Beta':<30}{weighted_sector_beta:>12.2f}")
        
        # New features
        print(f"  {'Downside Volatility':<30}{port_downside_vol*100:>11.2f}%")
        print(f"  {'Max Drawdown':<30}{port_max_drawdown*100:>11.2f}%")
        print(f"  {'Amihud Illiquidity':<30}{port_amihud:>12.6f}")
        print(f"  {'Debt to Equity':<30}{port_debt_to_equity:>12.3f}")
        print(f"  {'Revenue Growth Vol':<30}{port_revenue_growth_vol:>12.3f}")
        print(f"  {'Current Ratio':<30}{port_current_ratio:>12.3f}")
        print(f"  {'Interest Coverage':<30}{port_interest_coverage:>12.3f}")
        print(f"  {'Return on Assets (ROA)':<30}{port_roa:>12.3f}")

        print(f"\n  {'─'*42}")
        print(f"  {'Risk Score (Math)':<30}{score_result['Final_Risk_Score']:>12}")
        print(f"  {'MATH CLASSIFICATION':<30}{score_result['Risk_Category']:>12}")
        print(f"  {'AI CLASSIFICATION':<30}{ai_category:>12}")
        print(f"  {'AI Confidence':<15} {prob_str}")

        print(f"\n  SECTOR BREAKDOWN")
        print(f"  {'─'*42}")
        for sector_name, sw in portfolio_sectors.items():
            print(f"  {sector_name:<25}{sw*100:>6.1f}%")

        details = score_result.get('Details', {})
        if details:
            print(f"\n  Detailed Math Breakdown:")
            print(f"  {'─'*42}")
            print(f"  {'Norm. Volatility Score':<30}{details.get('Normalized_Port_Vol','N/A'):>12}")
            print(f"  {'Norm. Beta Score':<30}{details.get('Normalized_Port_Beta','N/A'):>12}")
            print(f"  {'Raw Sector Risk':<30}{details.get('Raw_Sector_Risk','N/A'):>12}")
            print(f"  {'Norm. Sector Risk':<30}{details.get('Normalized_Sector_Risk','N/A'):>12}")

        print("\n" + "★" * 45 + "\n")

    except Exception as e:
        print(f"\n  [Error] Could not calculate risk: {e}")
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 45)
    print("   TADAWUL PORTFOLIO RISK ANALYZER")
    print("   Powered by Math + AI (Rolling Window)")
    print("═" * 45)

    # اختيار المودل مرة واحدة في البداية
    model_info = select_model()
    model, scaler, encoder, features_to_use = load_model_artifacts(model_info)

    if model is None:
        sys.exit(1)

    while True:
        tickers, weights = get_user_portfolio()

        if tickers and len(tickers) > 0:
            process_prediction(
                tickers, weights,
                model, scaler, encoder, features_to_use, model_info
            )

        again = input("Analyze another portfolio? (y/n): ").strip().lower()
        if again != 'y':
            break

    print("\n  Goodbye!\n")
