import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import time

# TensorFlow is optional — only required for the LSTM model.
# On environments where TF is not installed (e.g. Streamlit Cloud free tier),
# RF and SVM still work normally; LSTM shows a clear unavailable message.
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# === Fix Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

try:
    from src.data_loader  import TadawulDataLoader
    from src.calculations import RiskCalculator
    from src.risk_labeler import RiskLabeler
except ModuleNotFoundError:
    st.error("⚠️ Error: 'src' folder not found. Make sure app.py is next to the src folder.")
    st.stop()

# === UI Configuration ===
st.set_page_config(page_title="Riskless Asset Management", page_icon="📈", layout="centered")

# ============================================================
# LOGO INJECTION
# ============================================================
logo_path = os.path.join(BASE_DIR, "logo.png")
if os.path.exists(logo_path):
    import base64
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div style="display:flex;justify-content:center;align-items:center;margin-bottom:1rem;">'
        f'<img src="data:image/png;base64,{logo_b64}" width="250"/>'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Please save your image as 'logo.png' in the same folder to see it here.")


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

# Original 6 features (always used)
ORIGINAL_FEATURES = [
    'Portfolio_Volatility',
    'Portfolio_Beta',
    'Sector_Volatility',
    'Sector_Beta',
    'Diversification_Index',
    'Market_Cap_Score',
]

# New portfolio-level features added from the expanded ML feature set
NEW_FEATURES = [
    'Portfolio_Downside_Volatility',
    'Portfolio_Max_Drawdown',
    'Portfolio_Amihud_Illiquidity',
    'Portfolio_Debt_to_Equity',
    'Portfolio_Revenue_Growth_Vol',
    'Portfolio_Current_Ratio',
    'Portfolio_Interest_Coverage',
    'Portfolio_ROA',
]

ALL_FEATURES = ORIGINAL_FEATURES + NEW_FEATURES

# Directional sets for normalization at inference time
POSITIVE_RISK_FEATURES = {
    'Portfolio_Volatility', 'Portfolio_Beta',
    'Sector_Volatility', 'Sector_Beta',
    'Portfolio_Downside_Volatility', 'Portfolio_Max_Drawdown',
    'Portfolio_Amihud_Illiquidity', 'Portfolio_Debt_to_Equity',
    'Portfolio_Revenue_Growth_Vol',
}
NEGATIVE_RISK_FEATURES = {
    'Diversification_Index', 'Market_Cap_Score',
    'Portfolio_Current_Ratio', 'Portfolio_Interest_Coverage',
    'Portfolio_ROA',
}


# ============================================================
# MODEL REGISTRY — RF + LSTM + SVM
# ============================================================

MODELS = {
    "Random Forest": {
        "type"    : "rf",
        "model"   : "rf_rolling_window.pkl",
        "scaler"  : None,                          # RF doesn't need scaling
        "encoder" : None,                          # RF uses string labels directly
        "features": "rf_features.pkl",             # saved feature list
        "color"   : "#f59e0b",
        "icon"    : "🌲",
        "desc"    : "Random Forest · Rolling Window",
    },
    "LSTM": {
        "type"    : "lstm",
        "model"   : "lstm_rolling_window.keras",
        "scaler"  : "lstm_scaler.pkl",
        "encoder" : "lstm_label_encoder.pkl",
        "features": "lstm_features.pkl",
        "color"   : "#a78bfa",
        "icon"    : "🧠",
        "desc"    : "LSTM Neural Network · Rolling Window",
    },
    "SVM": {
        "type"    : "svm",
        "model"   : "svm_rolling_window.pkl",
        "scaler"  : "svm_scaler.pkl",
        "encoder" : "svm_label_encoder.pkl",
        "features": "svm_features.pkl",
        "color"   : "#34d399",
        "icon"    : "⚡",
        "desc"    : "Support Vector Machine · Rolling Window",
    },
}


# ============================================================
# CACHING LAYER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_artifacts(model_name: str):
    """Load model + scaler + encoder + feature list for the selected model."""
    info       = MODELS[model_name]
    models_dir = os.path.join(BASE_DIR, "models")
    model_path = os.path.join(models_dir, info["model"])

    if not os.path.exists(model_path):
        return None, None, None, None

    # ── Model ──────────────────────────────────────────────
    if info["type"] == "lstm":
        if not TF_AVAILABLE:
            return None, None, None, None   # signal: TF not installed
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)

    # ── Scaler ─────────────────────────────────────────────
    scaler = None
    if info["scaler"]:
        scaler_path = os.path.join(models_dir, info["scaler"])
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)

    # ── Encoder ────────────────────────────────────────────
    encoder = None
    if info["encoder"]:
        encoder_path = os.path.join(models_dir, info["encoder"])
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)

    # ── Feature list saved during training ─────────────────
    features = ALL_FEATURES          # default: use full list
    if info["features"]:
        feat_path = os.path.join(models_dir, info["features"])
        if os.path.exists(feat_path):
            features = joblib.load(feat_path)

    return model, scaler, encoder, features


@st.cache_resource(show_spinner=False)
def load_metadata():
    meta_path = os.path.join(BASE_DIR, 'data', 'raw', "stocks_metadata.csv")
    if os.path.exists(meta_path):
        return pd.read_csv(meta_path).set_index("Ticker")
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_and_calculate(tickers_tuple, weights_tuple):
    tickers = list(tickers_tuple)
    weights = list(weights_tuple)

    # ── Use a per-portfolio temp directory so each ticker combination
    #    gets its own isolated CSV files — no stale data from other runs.
    safe_key       = "_".join(sorted(tickers))
    data_directory = os.path.join(BASE_DIR, 'data', 'live', safe_key)
    os.makedirs(data_directory, exist_ok=True)

    stocks_csv = os.path.join(data_directory, "stocks_prices.csv")
    market_csv = os.path.join(data_directory, "market_prices.csv")
    meta_path  = os.path.join(data_directory, "stocks_metadata.csv")

    # Download only the user-selected tickers (not the full 20-ticker universe)
    # This avoids geo-blocking issues on Streamlit Cloud for unused tickers
    # and removes the universe size mismatch error entirely.
    loader = TadawulDataLoader(tickers=tickers, data_dir=data_directory)

    if not os.path.exists(stocks_csv):
        loader.fetch_stock_data()
    if not os.path.exists(market_csv):
        loader.fetch_market_data()
    if not os.path.exists(meta_path):
        loader.fetch_metadata()

    # Guard: if download failed (e.g. geo-blocked), raise a clear message
    for csv_path, label in [(stocks_csv, "stocks_prices.csv"),
                             (market_csv, "market_prices.csv")]:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 50:
            raise ConnectionError(
                f"Could not download market data ({label}). "
                f"Yahoo Finance may be temporarily unavailable. "
                f"Please try again in a few seconds."
            )

    meta_df = pd.read_csv(meta_path).set_index("Ticker")

    # RiskCalculator now loads only the selected tickers' prices —
    # weights vector length matches exactly (no universe alignment needed)
    calc = RiskCalculator(data_dir=data_directory)
    calc.load_data()
    calc.calculate_daily_returns()

    # weights aligns 1-to-1 with tickers — no full_weights padding needed
    # because calc.tickers == the user's selected tickers
    metrics = calc.calculate_portfolio_risk(weights)
    vol     = metrics['Portfolio_Volatility_Percentage']
    beta    = metrics['Portfolio_Beta']

    div_index = 1.0 - np.sum(np.array(weights) ** 2)

    portfolio_sectors = {}
    port_cap_score    = 0.0

    for t, w in zip(tickers, weights):
        score  = meta_df.loc[t, "Market_Cap_Score"] if t in meta_df.index else 2.0
        port_cap_score += w * score
        sector = meta_df.loc[t, "Sector"] if (
            t in meta_df.index and "Sector" in meta_df.columns
        ) else loader.sector_map.get(t, "Unknown")
        portfolio_sectors[sector] = portfolio_sectors.get(sector, 0.0) + w

    weighted_sector_vol  = 0.0
    weighted_sector_beta = 0.0

    for sec, sec_weight in portfolio_sectors.items():
        sec_tickers = [tk for tk, s in loader.sector_map.items() if s == sec]
        # Filter to only tickers present in calc (the user's selection)
        sec_tickers_available = [tk for tk in sec_tickers if tk in calc.tickers]
        if sec_tickers_available:
            s_vol, s_beta = calc.calculate_sector_metrics(sec_tickers_available)
        else:
            s_vol, s_beta = 0.15, 1.0
        weighted_sector_vol  += sec_weight * s_vol
        weighted_sector_beta += sec_weight * s_beta

    new_feature_dict = {}

    new_feature_dict['Portfolio_Downside_Volatility'] = _weighted_stock_metric(
        calc, tickers, weights, 'Downside_Volatility'
    )
    new_feature_dict['Portfolio_Max_Drawdown'] = _weighted_stock_metric(
        calc, tickers, weights, 'Max_Drawdown'
    )
    new_feature_dict['Portfolio_Amihud_Illiquidity'] = _weighted_stock_metric(
        calc, tickers, weights, 'Amihud_Illiquidity'
    )

    new_feature_dict['Portfolio_Debt_to_Equity'] = _weighted_meta_metric(
        meta_df, tickers, weights, 'Debt_to_Equity'
    )
    new_feature_dict['Portfolio_Revenue_Growth_Vol'] = _weighted_meta_metric(
        meta_df, tickers, weights, 'Revenue_Growth_Vol'
    )
    new_feature_dict['Portfolio_Current_Ratio'] = _weighted_meta_metric(
        meta_df, tickers, weights, 'Current_Ratio'
    )
    new_feature_dict['Portfolio_Interest_Coverage'] = _weighted_meta_metric(
        meta_df, tickers, weights, 'Interest_Coverage'
    )
    new_feature_dict['Portfolio_ROA'] = _weighted_meta_metric(
        meta_df, tickers, weights, 'ROA'
    )

    labeler      = RiskLabeler()
    score_result = labeler.calculate_final_score(
        port_q_pct=vol, port_b=beta,
        sector_q=weighted_sector_vol, sector_b=weighted_sector_beta
    )

    return {
        'vol'                 : vol,
        'beta'                : beta,
        'div_index'           : div_index,
        'port_cap_score'      : port_cap_score,
        'portfolio_sectors'   : portfolio_sectors,
        'weighted_sector_vol' : weighted_sector_vol,
        'weighted_sector_beta': weighted_sector_beta,
        'score_result'        : score_result,
        'meta_df'             : meta_df,
        'sector_map'          : loader.sector_map,
        'new_features'        : new_feature_dict,
    }


def _weighted_stock_metric(calc, tickers, weights, metric_name):
    """
    Compute portfolio-weighted average of a per-stock metric from RiskCalculator.
    Returns NaN if the metric is not available.
    """
    total = 0.0
    total_weight = 0.0
    for t, w in zip(tickers, weights):
        try:
            val = calc.get_stock_metric(t, metric_name)
            if val is not None and not np.isnan(val):
                total        += w * val
                total_weight += w
        except Exception:
            pass
    return (total / total_weight) if total_weight > 0 else np.nan


def _weighted_meta_metric(meta_df, tickers, weights, col_name):
    """
    Compute portfolio-weighted average of a fundamental metric from metadata CSV.
    Returns NaN if the column is missing or all tickers lack the value.
    """
    if meta_df is None or col_name not in meta_df.columns:
        return np.nan
    total = 0.0
    total_weight = 0.0
    for t, w in zip(tickers, weights):
        if t in meta_df.index:
            val = meta_df.loc[t, col_name]
            try:
                val = float(val)
                if not np.isnan(val):
                    total        += w * val
                    total_weight += w
            except (TypeError, ValueError):
                pass
    return (total / total_weight) if total_weight > 0 else np.nan


def build_feature_vector(results, features_to_use):
    """
    Assemble and return the ordered feature vector for model inference.
    NaN values (features not available) are median-imputed with 0.5
    (post-normalization midpoint — conservative safe default).
    """
    vol                  = results['vol']
    beta                 = results['beta']
    div_index            = results['div_index']
    port_cap_score       = results['port_cap_score']
    weighted_sector_vol  = results['weighted_sector_vol']
    weighted_sector_beta = results['weighted_sector_beta']
    new_features         = results['new_features']

    base_values = {
        'Portfolio_Volatility'  : vol,
        'Portfolio_Beta'        : beta,
        'Sector_Volatility'     : weighted_sector_vol * 100,
        'Sector_Beta'           : weighted_sector_beta,
        'Diversification_Index' : div_index,
        'Market_Cap_Score'      : port_cap_score,
    }
    all_values = {**base_values, **new_features}

    # Build ordered list; impute missing with 0.5 (mid-range safe default)
    feature_vector = []
    imputed = []
    for feat in features_to_use:
        val = all_values.get(feat, np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.5
            imputed.append(feat)
        feature_vector.append(val)

    if imputed:
        st.caption(f"ℹ️ Features imputed with 0.5 (not available in live data): {', '.join(imputed)}")

    return feature_vector


def ai_predict(model_name, model, scaler, encoder, features_to_use, results):
    """Run inference for the selected model."""
    mtype          = MODELS[model_name]["type"]
    feature_vector = build_feature_vector(results, features_to_use)
    X_raw          = np.array(feature_vector).reshape(1, -1)

    if mtype == "rf":
        # RF: no scaling needed; predict directly (string labels)
        ai_category = model.predict(X_raw)[0]
        if hasattr(model, "predict_proba"):
            probs     = model.predict_proba(X_raw)[0]
            classes   = model.classes_
            prob_dict = {c: round(float(p) * 100) for c, p in zip(classes, probs)}
        else:
            prob_dict = {}

    elif mtype == "lstm":
        if not TF_AVAILABLE:
            return "Unavailable", {}
        X_scaled = scaler.transform(X_raw) if scaler else X_raw
        X_3d     = X_scaled.reshape(1, 1, X_scaled.shape[1])
        probs    = model.predict(X_3d, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        ai_category = encoder.inverse_transform([pred_idx])[0] if encoder else str(pred_idx)
        classes   = encoder.classes_ if encoder else list(range(len(probs)))
        prob_dict = {c: round(float(p) * 100) for c, p in zip(classes, probs)}

    elif mtype == "svm":
        X_scaled     = scaler.transform(X_raw) if scaler else X_raw
        pred_encoded = model.predict(X_scaled)[0]
        ai_category  = encoder.inverse_transform([pred_encoded])[0] if encoder else str(pred_encoded)
        if hasattr(model, "predict_proba"):
            probs     = model.predict_proba(X_scaled)[0]
            classes   = encoder.classes_ if encoder else model.classes_
            prob_dict = {c: round(float(p) * 100) for c, p in zip(classes, probs)}
        else:
            prob_dict = {}

    else:
        ai_category = "Unknown"
        prob_dict   = {}

    return ai_category, prob_dict


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1b2838 100%);
    }
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Hero ── */
    .hero-container {
        background: linear-gradient(135deg, rgba(30,60,114,0.4), rgba(42,82,152,0.3));
        border: 1px solid rgba(100,150,255,0.15);
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(100,150,255,0.05) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { transform: scale(1); opacity: .5; }
        50%      { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: clamp(1.4rem, 5vw, 2.8rem);
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #60a5fa);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        margin-bottom: .5rem;
        position: relative; z-index: 1;
        line-height: 1.2;
    }
    @keyframes shimmer {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        color: rgba(200,210,230,0.8);
        font-size: clamp(0.85rem, 2.5vw, 1.1rem);
        font-weight: 300;
        position: relative; z-index: 1;
        letter-spacing: .3px;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(96,165,250,0.2), rgba(167,139,250,0.2));
        border: 1px solid rgba(96,165,250,0.3);
        border-radius: 50px;
        padding: .3rem .8rem;
        font-size: clamp(0.7rem, 2vw, 0.8rem);
        color: #93c5fd;
        margin-top: .8rem;
        position: relative; z-index: 1;
        font-family: 'Inter', sans-serif;
    }

    /* ── Model Selector Cards ── */
    .model-selector-row {
        display: flex;
        gap: 10px;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .model-pill {
        flex: 1 1 100px;
        background: rgba(30,41,59,.5);
        border: 1px solid rgba(100,150,255,.15);
        border-radius: 12px;
        padding: .7rem .5rem;
        text-align: center;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: all .2s ease;
    }
    .model-pill.active {
        border-width: 2px;
        background: rgba(30,41,59,.8);
    }
    .model-pill-icon  { font-size: 1.4rem; }
    .model-pill-name  { font-size: .8rem; font-weight: 700; color: #e2e8f0; margin-top: .3rem; }
    .model-pill-desc  { font-size: .62rem; color: rgba(200,210,230,.45); margin-top: .15rem; }

    /* ── Result Cards ── */
    .result-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.6), rgba(30,41,59,0.3));
        border: 1px solid rgba(100,150,255,0.12);
        border-radius: 16px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        transition: all .3s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .result-card:hover {
        border-color: rgba(100,150,255,0.3);
        box-shadow: 0 8px 30px rgba(0,0,0,.3);
    }
    .result-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .result-card.math-card::after  { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
    .result-card.ai-card-rf::after   { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
    .result-card.ai-card-lstm::after { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
    .result-card.ai-card-svm::after  { background: linear-gradient(90deg,#10b981,#34d399); }

    .card-label { font-family:'Inter',sans-serif; color:rgba(200,210,230,.6); font-size:.78rem; text-transform:uppercase; letter-spacing:1.5px; font-weight:500; margin-bottom:.3rem; }
    .card-title { font-family:'Inter',sans-serif; color:#e2e8f0; font-size:1.1rem; font-weight:700; margin-bottom:1rem; }

    /* ── Score Circle ── */
    .score-circle {
        width: clamp(100px, 25vw, 130px);
        height: clamp(100px, 25vw, 130px);
        border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        flex-direction:column;
        margin:.8rem auto;
    }
    .score-circle.low    { background:radial-gradient(circle,rgba(34,197,94,.15),transparent 70%);  border:3px solid rgba(34,197,94,.4);  box-shadow:0 0 25px rgba(34,197,94,.15); }
    .score-circle.medium { background:radial-gradient(circle,rgba(234,179,8,.15),transparent 70%);  border:3px solid rgba(234,179,8,.4);  box-shadow:0 0 25px rgba(234,179,8,.15); }
    .score-circle.high   { background:radial-gradient(circle,rgba(239,68,68,.15),transparent 70%);  border:3px solid rgba(239,68,68,.4);  box-shadow:0 0 25px rgba(239,68,68,.15); }

    .score-value        { font-family:'Inter',sans-serif; font-size:clamp(1.6rem, 5vw, 2.4rem); font-weight:800; }
    .score-value.low    { color:#22c55e; }
    .score-value.medium { color:#eab308; }
    .score-value.high   { color:#ef4444; }
    .score-label-small  { font-family:'Inter',sans-serif; font-size:.65rem; color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:1px; }

    /* ── Risk Badge ── */
    .risk-badge        { display:inline-block; padding:.4rem 1.2rem; border-radius:50px; font-family:'Inter',sans-serif; font-weight:600; font-size:.88rem; text-align:center; margin-top:.5rem; }
    .risk-badge.low    { background:rgba(34,197,94,.15);  border:1px solid rgba(34,197,94,.4);  color:#4ade80; }
    .risk-badge.medium { background:rgba(234,179,8,.15);  border:1px solid rgba(234,179,8,.4);  color:#facc15; }
    .risk-badge.high   { background:rgba(239,68,68,.15);  border:1px solid rgba(239,68,68,.4);  color:#f87171; }

    /* ── Metric Cards ── */
    .metrics-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:1rem; }
    .metric-card  { background:linear-gradient(135deg,rgba(30,41,59,.5),rgba(30,41,59,.2)); border:1px solid rgba(100,150,255,.1); border-radius:12px; padding:1rem .6rem; text-align:center; transition:all .3s ease; height:100%; }
    .metric-card:hover { border-color:rgba(100,150,255,.25); }
    .metric-icon  { font-size:1.4rem; margin-bottom:.3rem; }
    .metric-value { font-family:'Inter',sans-serif; font-size:clamp(1rem, 3.5vw, 1.6rem); font-weight:700; color:#e2e8f0; }
    .metric-name  { font-family:'Inter',sans-serif; font-size:clamp(0.6rem, 1.8vw, 0.75rem); color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:.8px; margin-top:.3rem; }

    /* ── Section ── */
    .section-header  { font-family:'Inter',sans-serif; color:#e2e8f0; font-size:clamp(1.1rem, 3vw, 1.4rem); font-weight:700; margin:1.5rem 0 .8rem 0; display:flex; align-items:center; gap:.5rem; }
    .section-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(100,150,255,.2),transparent); margin:1.2rem 0; }

    /* ── Holdings Table ── */
    .holdings-container { background:linear-gradient(135deg,rgba(30,41,59,.5),rgba(30,41,59,.2)); border:1px solid rgba(100,150,255,.1); border-radius:12px; padding:1rem; margin-top:.8rem; overflow-x:auto; -webkit-overflow-scrolling:touch; }
    .sector-table       { width:100%; min-width:360px; border-collapse:separate; border-spacing:0; border-radius:10px; overflow:hidden; font-family:'Inter',sans-serif; }
    .sector-table thead th { background:rgba(30,41,59,.8); color:#93c5fd; padding:.6rem .8rem; font-size:.72rem; text-transform:uppercase; letter-spacing:1px; font-weight:600; border-bottom:1px solid rgba(100,150,255,.15); white-space:nowrap; }
    .sector-table tbody td { padding:.6rem .8rem; color:#cbd5e1; font-size:.82rem; border-bottom:1px solid rgba(100,150,255,.05); background:rgba(15,23,42,.3); }
    .sector-table tbody tr:hover td { background:rgba(30,41,59,.5); }

    /* ── Model Comparison Table ── */
    .compare-table { width:100%; border-collapse:separate; border-spacing:0; border-radius:10px; overflow:hidden; font-family:'Inter',sans-serif; margin-top:.6rem; }
    .compare-table thead th { background:rgba(30,41,59,.8); color:#93c5fd; padding:.6rem .8rem; font-size:.72rem; text-transform:uppercase; letter-spacing:1px; font-weight:600; border-bottom:1px solid rgba(100,150,255,.15); white-space:nowrap; }
    .compare-table tbody td { padding:.6rem .8rem; color:#cbd5e1; font-size:.82rem; border-bottom:1px solid rgba(100,150,255,.05); background:rgba(15,23,42,.3); text-align:center; }
    .compare-table tbody tr:hover td { background:rgba(30,41,59,.5); }
    .compare-table tbody td:first-child { text-align:left; font-weight:600; }

    /* ── Probability Bars ── */
    .prob-section       { padding:.3rem 0; }
    .prob-section-title { font-family:'Inter',sans-serif; font-size:.68rem; font-weight:500; color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:.08em; margin-bottom:.8rem; }
    .prob-row           { margin-bottom:12px; }
    .prob-meta          { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; }
    .prob-label         { font-family:'Inter',sans-serif; font-size:.82rem; color:#cbd5e1; font-weight:500; display:flex; align-items:center; gap:7px; }
    .prob-dot           { width:7px; height:7px; border-radius:50%; flex-shrink:0; display:inline-block; }
    .prob-dot.low    { background:#22c55e; }
    .prob-dot.medium { background:#eab308; }
    .prob-dot.high   { background:#ef4444; }
    .prob-pct        { font-family:'Inter',sans-serif; font-size:.88rem; font-weight:600; }
    .prob-pct.low    { color:#4ade80; }
    .prob-pct.medium { color:#facc15; }
    .prob-pct.high   { color:#f87171; }
    .prob-track      { height:6px; background:rgba(30,41,59,.8); border-radius:3px; overflow:hidden; }
    .prob-fill       { height:100%; border-radius:3px; }
    .prob-fill.low    { background:linear-gradient(90deg,#22c55e,#4ade80); }
    .prob-fill.medium { background:linear-gradient(90deg,#eab308,#facc15); }
    .prob-fill.high   { background:linear-gradient(90deg,#ef4444,#f87171); }
    .prob-sublabel    { font-family:'Inter',sans-serif; font-size:.68rem; color:rgba(200,210,230,.35); margin-top:2px; }

    /* ── Sector Grid ── */
    .sector-grid { display:flex; flex-wrap:wrap; gap:10px; margin-top:.5rem; }
    .sector-card { flex:1 1 120px; min-width:100px; background:linear-gradient(135deg,rgba(30,41,59,.5),rgba(30,41,59,.2)); border:1px solid rgba(100,150,255,.1); border-radius:12px; padding:1rem .6rem; text-align:center; }
    .sector-pct  { font-family:'Inter',sans-serif; font-size:1.4rem; font-weight:700; }
    .sector-name { font-family:'Inter',sans-serif; font-size:.65rem; color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:.8px; margin-top:.3rem; }

    /* ── Buttons ── */
    .stButton > button { background:linear-gradient(135deg,#3b82f6,#8b5cf6) !important; color:white !important; border:none !important; border-radius:12px !important; padding:.75rem 1.5rem !important; font-family:'Inter',sans-serif !important; font-weight:600 !important; font-size:1rem !important; transition:all .3s ease !important; box-shadow:0 4px 15px rgba(59,130,246,.3) !important; min-height:48px !important; }
    .stButton > button:active { transform:scale(0.97) !important; }

    /* ── Banners ── */
    .success-banner { background:linear-gradient(135deg,rgba(34,197,94,.15),rgba(34,197,94,.05)); border:1px solid rgba(34,197,94,.3); border-radius:12px; padding:.8rem 1rem; display:flex; align-items:center; gap:.6rem; font-family:'Inter',sans-serif; color:#4ade80; font-weight:500; margin-bottom:1.2rem; flex-wrap:wrap; font-size:clamp(.82rem,2.5vw,.95rem); }
    .speed-chip     { display:inline-flex; align-items:center; gap:.4rem; background:rgba(139,92,246,.1); border:1px solid rgba(139,92,246,.25); border-radius:50px; padding:.25rem .7rem; font-family:'Inter',sans-serif; font-size:.72rem; color:#a78bfa; }

    [data-testid="column"] { padding-left:.3rem !important; padding-right:.3rem !important; }
    input[type="number"], input[type="text"] { min-height:44px; font-size:16px !important; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
    [data-testid="collapsedControl"] { display:none !important; }
    section[data-testid="stSidebar"]  { display:none !important; }

    @media (max-width: 640px) {
        .block-container  { padding-left:.75rem !important; padding-right:.75rem !important; }
        .hero-container   { padding:1.2rem 1rem; border-radius:12px; margin-bottom:1rem; }
        .result-card      { padding:1rem; border-radius:12px; margin-bottom:.8rem; }
        .score-circle     { width:90px !important; height:90px !important; }
        .metrics-grid     { grid-template-columns:repeat(2,1fr); gap:8px; }
        .metric-card      { padding:.8rem .5rem; }
        .metric-value     { font-size:1.2rem; }
        .metric-name      { font-size:.6rem; }
        .section-header   { font-size:1rem; margin:1rem 0 .6rem 0; }
        .prob-sublabel    { display:none; }
        .sector-table thead th { font-size:.65rem; padding:.5rem; }
        .sector-table tbody td { font-size:.75rem; padding:.5rem; }
        .success-banner   { padding:.7rem .9rem; gap:.5rem; }
        .model-pill-desc  { display:none; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-title">📈 Tadawul Portfolio Risk Analyzer</div>
    <div class="hero-subtitle">
        Predict the risk of any Saudi Stock Market portfolio using <strong>Mathematics</strong> &amp; <strong>Artificial Intelligence</strong>
    </div>
    <div class="hero-badge">🔬 Powered by Machine Learning · Rolling Window Validation · 14-Feature ML Set</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL SELECTOR
# ============================================================

st.markdown('<div class="section-header">🤖 Select AI Model</div>', unsafe_allow_html=True)

model_options = list(MODELS.keys())
selected_model = st.radio(
    "AI Model",
    model_options,
    horizontal=True,
    label_visibility="collapsed",
)

info = MODELS[selected_model]
st.markdown(
    f'<div style="background:rgba(30,41,59,.5);border:1px solid {info["color"]}40;border-radius:10px;'
    f'padding:.6rem 1rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:.6rem;">'
    f'<span style="font-size:1.3rem;">{info["icon"]}</span>'
    f'<span style="font-family:Inter,sans-serif;color:{info["color"]};font-weight:600;font-size:.9rem;">'
    f'{selected_model}</span>'
    f'<span style="font-family:Inter,sans-serif;color:rgba(200,210,230,.45);font-size:.8rem;">· {info["desc"]}</span>'
    f'</div>',
    unsafe_allow_html=True
)

# Preload selected model silently
model, scaler, encoder, features_to_use = load_model_artifacts(selected_model)
if info["type"] == "lstm" and not TF_AVAILABLE:
    st.warning(
        "⚠️ LSTM is not available on this deployment — TensorFlow is not installed. "
        "Select **Random Forest** or **SVM** instead. "
        "To use LSTM, run the app locally with `tensorflow-cpu` installed."
    )
elif model is None:
    st.markdown(
        '<div style="background:rgba(234,179,8,.08);border:1px solid rgba(234,179,8,.3);' +
        'border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1rem;">' +
        '<div style="font-family:Inter,sans-serif;color:#facc15;font-weight:700;font-size:1rem;margin-bottom:.8rem;">' +
        '&#9888; Models not found — follow these steps to get started' +
        '</div>' +
        '<div style="font-family:Inter,sans-serif;color:rgba(200,210,230,.75);font-size:.88rem;line-height:2;">' +
        '<b style="color:#e2e8f0;">Step 1</b> &nbsp; Train the models locally:' +
        '<br><code style="background:rgba(0,0,0,.3);padding:3px 8px;border-radius:5px;font-size:.82rem;">' +
        'cd src &nbsp;&&nbsp; python data_generator.py' +
        '</code>' +
        '<br><code style="background:rgba(0,0,0,.3);padding:3px 8px;border-radius:5px;font-size:.82rem;">' +
        'python ml_model_rf.py &nbsp;&&nbsp; python ml_model_svm.py' +
        '</code>' +
        '<br><br><b style="color:#e2e8f0;">Step 2</b> &nbsp; Commit the generated model files:' +
        '<br><code style="background:rgba(0,0,0,.3);padding:3px 8px;border-radius:5px;font-size:.82rem;">' +
        'git add models/ &nbsp;&&nbsp; git commit -m \"add trained models\" &nbsp;&&nbsp; git push' +
        '</code>' +
        '<br><br><b style="color:#e2e8f0;">Step 3</b> &nbsp; Streamlit Cloud will redeploy automatically.' +
        '</div></div>',
        unsafe_allow_html=True
    )


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ============================================================
# PORTFOLIO BUILDER
# ============================================================

with st.expander("💼 Portfolio Builder — tap to configure", expanded=True):

    num_stocks = st.number_input(
        "Number of Stocks", min_value=1, max_value=10, value=1,
        help="Select how many stocks to include in your portfolio"
    )

    tickers = []
    weights = []

    for i in range(num_stocks):
        st.markdown(
            f'<div style="font-family:Inter,sans-serif;color:#93c5fd;font-size:.82rem;'
            f'font-weight:600;margin:.6rem 0 .2rem;letter-spacing:.5px;">STOCK {i+1}</div>',
            unsafe_allow_html=True
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input(
                "Ticker", value="2222.SR" if i == 0 else "",
                key=f"t_{i}", placeholder="e.g., 2222.SR",
                label_visibility="collapsed"
            )
        with col2:
            weight = st.number_input(
                "Weight %",
                min_value=1.0, max_value=100.0,
                value=round(100.0 / num_stocks, 1),
                key=f"w_{i}",
                label_visibility="collapsed"
            )

        if ticker:
            t_upper = ticker.upper().strip()
            if not t_upper.endswith('.SR'):
                t_upper += '.SR'
            tickers.append(t_upper)
            weights.append(weight / 100.0)

    total_weight  = sum(weights) * 100 if weights else 0
    is_valid      = abs(total_weight - 100) < 1
    weight_color  = "#4ade80" if is_valid else "#f87171"
    check_icon    = "✓" if is_valid else "✗"
    border_color  = "rgba(34,197,94,.3)" if is_valid else "rgba(239,68,68,.3)"

    st.markdown(
        f'<div style="background:rgba(30,41,59,.5);border:1px solid {border_color};'
        f'border-radius:10px;padding:.7rem;text-align:center;margin:.8rem 0;">'
        f'<span style="color:rgba(200,210,230,.5);font-size:.72rem;text-transform:uppercase;letter-spacing:1px;">'
        f'Total Weight {check_icon}</span><br>'
        f'<span style="color:{weight_color};font-size:1.4rem;font-weight:700;">{total_weight:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    analyze_button = st.button("🚀 Analyze Portfolio Risk", use_container_width=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_class(category):
    cat = str(category)
    if "Low"    in cat: return "low"
    if "Medium" in cat: return "medium"
    return "high"

def get_risk_emoji(category):
    cat = str(category)
    if "Low"    in cat: return "🟢"
    if "Medium" in cat: return "🟡"
    return "🔴"

def get_risk_description(category):
    cat = str(category)
    if "Low"    in cat: return "This portfolio shows conservative risk characteristics. Suitable for risk-averse investors."
    if "Medium" in cat: return "This portfolio has moderate risk exposure. A balanced approach for most investors."
    return "This portfolio exhibits high risk levels. Only suitable for aggressive, experienced investors."

def get_prob_sublabel(cat):
    cat = str(cat)
    if "Low"    in cat: return "High confidence — conservative portfolio characteristics"
    if "Medium" in cat: return "Marginal — moderate exposure detected"
    return "Elevated — aggressive risk signals present"

def build_prob_bars_html(prob_dict):
    bars_html = ""
    if prob_dict:
        for cat, pct in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
            bar_class = get_risk_class(cat)
            sublabel  = get_prob_sublabel(cat)
            bars_html += (
                '<div class="prob-row">'
                '<div class="prob-meta">'
                '<span class="prob-label">'
                f'<span class="prob-dot {bar_class}"></span>{cat}'
                '</span>'
                f'<span class="prob-pct {bar_class}">{pct}%</span>'
                '</div>'
                f'<div class="prob-track"><div class="prob-fill {bar_class}" style="width:{pct}%;"></div></div>'
                f'<div class="prob-sublabel">{sublabel}</div>'
                '</div>'
            )
        return (
            '<div style="margin-top:1.2rem;padding:1rem;background:rgba(15,23,42,.3);border-radius:10px;">'
            '<div class="prob-section-title">Probability Distribution</div>'
            f'<div class="prob-section">{bars_html}</div>'
            '</div>'
        )
    return ""


# ============================================================
# MAIN ANALYSIS LOGIC
# ============================================================

if analyze_button:
    if abs(sum(weights) - 1.0) > 0.01:
        st.error("⚠️ Total weights must equal 100%!")
    elif len(tickers) == 0:
        st.error("⚠️ Please enter at least one stock.")
    elif info["type"] == "lstm" and not TF_AVAILABLE:
        st.error(
            "⚠️ LSTM requires TensorFlow which is not available on this deployment. "
            "Please select **Random Forest** or **SVM**."
        )
    elif model is None:
        st.markdown(
            '<div style="background:rgba(234,179,8,.08);border:1px solid rgba(234,179,8,.3);' +
            'border-radius:14px;padding:1.2rem 1.4rem;">' +
            '<div style="font-family:Inter,sans-serif;color:#facc15;font-weight:700;margin-bottom:.5rem;">' +
            '&#9888; Model not trained yet' +
            '</div>' +
            '<div style="font-family:Inter,sans-serif;color:rgba(200,210,230,.7);font-size:.85rem;line-height:1.9;">' +
            f'Run <code>src/ml_model_{info["type"]}.py</code> locally, then ' +
            'commit the <code>models/</code> folder and push to GitHub.' +
            '</div></div>',
            unsafe_allow_html=True
        )
    else:
        start_time   = time.time()
        progress_bar = st.progress(0)
        status_text  = st.empty()

        try:
            status_text.markdown("""
            <div style="font-family:'Inter',sans-serif; color:#93c5fd; font-size:.95rem; padding:.5rem;">
                ⏳ Fetching market data from Tadawul...
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(15)

            results = fetch_and_calculate(tuple(tickers), tuple(weights))
            progress_bar.progress(60)

            vol                  = results['vol']
            beta                 = results['beta']
            div_index            = results['div_index']
            port_cap_score       = results['port_cap_score']
            portfolio_sectors    = results['portfolio_sectors']
            weighted_sector_vol  = results['weighted_sector_vol']
            weighted_sector_beta = results['weighted_sector_beta']
            score_result         = results['score_result']
            meta_df              = results['meta_df']
            sector_map           = results['sector_map']

            status_text.markdown(f"""
            <div style="font-family:'Inter',sans-serif; color:{info['color']}; font-size:.95rem; padding:.5rem;">
                {info['icon']} Running {selected_model} prediction...
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(80)

            # ── AI Prediction ───────────────────────────────
            ai_category, prob_dict = ai_predict(
                selected_model, model, scaler, encoder, features_to_use, results
            )

            # ── Run all models for comparison table ─────────
            all_model_results = {}
            for m_name in MODELS:
                try:
                    m_model, m_scaler, m_encoder, m_feats = load_model_artifacts(m_name)
                    if m_model is not None:
                        cat, probs = ai_predict(m_name, m_model, m_scaler, m_encoder, m_feats, results)
                        conf = probs.get(cat, 0) if probs else 0
                        all_model_results[m_name] = {'category': cat, 'confidence': conf}
                except Exception:
                    pass

            progress_bar.progress(100)
            elapsed = time.time() - start_time
            status_text.empty()
            progress_bar.empty()

            # ── Success Banner ──────────────────────────────
            speed_chip = '<span class="speed-chip">&#9889; Cached</span>' if elapsed < 2 else ''
            st.markdown(
                f'<div class="success-banner">'
                f'<span style="font-size:1.3rem;">&#9989;</span>'
                f'<span>Analysis Complete &#8212; {selected_model} results in <strong>{elapsed:.1f}s</strong></span>'
                f'{speed_chip}'
                f'</div>',
                unsafe_allow_html=True
            )

            math_class = get_risk_class(score_result['Risk_Category'])
            ai_class   = get_risk_class(ai_category)
            math_emoji = get_risk_emoji(score_result['Risk_Category'])
            ai_emoji   = get_risk_emoji(ai_category)
            ai_css     = f"ai-card-{info['type']}"

            col1, col2 = st.columns(2, gap="large")

            # ── Math Card ───────────────────────────────────
            with col1:
                risk_score = score_result['Final_Risk_Score']
                risk_desc = get_risk_description(score_result['Risk_Category'])
                st.markdown(
                    f'<div class="result-card math-card">'
                    f'<div class="card-label">Mathematical Model</div>'
                    f'<div class="card-title">&#129518; Quantitative Analysis</div>'
                    f'<div class="score-circle {math_class}">'
                    f'<div class="score-value {math_class}">{risk_score}</div>'
                    f'<div class="score-label-small">Risk Score</div>'
                    f'</div>'
                    f'<div style="text-align:center;margin-top:1rem;">'
                    f'<div class="risk-badge {math_class}">{math_emoji} {score_result["Risk_Category"]}</div>'
                    f'</div>'
                    f'<div style="margin-top:1.2rem;padding:.8rem;background:rgba(15,23,42,.3);border-radius:10px;'
                    f'font-family:Inter,sans-serif;color:rgba(200,210,230,.5);font-size:.8rem;text-align:center;">'
                    f'{risk_desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── AI Card ─────────────────────────────────────
            with col2:
                ai_confidence = prob_dict.get(ai_category, 0)
                prob_block    = build_prob_bars_html(prob_dict)

                st.markdown(
                    f'<div class="result-card {ai_css}">'
                    f'<div class="card-label">{info["desc"]}</div>'
                    f'<div class="card-title">{info["icon"]} AI Prediction</div>'
                    f'<div class="score-circle {ai_class}">'
                    f'<div class="score-value {ai_class}">{ai_confidence}%</div>'
                    f'<div class="score-label-small">Confidence</div>'
                    f'</div>'
                    f'<div style="text-align:center;margin-top:1rem;">'
                    f'<div class="risk-badge {ai_class}">{ai_emoji} {ai_category}</div>'
                    f'</div>'
                    f'{prob_block}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Model Comparison Table ───────────────────────
            if len(all_model_results) > 1:
                math_cat       = score_result['Risk_Category']
                math_emoji_str = get_risk_emoji(math_cat)

                # Build every row as a single-line string — no multiline f-strings
                # that Streamlit's sanitiser might split across markdown passes.
                rows_html = (
                    f'<tr>'
                    f'<td>🧮 Math Benchmark</td>'
                    f'<td>{risk_score}</td>'
                    f'<td>{math_emoji_str} {math_cat}</td>'
                    f'<td>—</td>'
                    f'</tr>'
                )
                for m_name, m_res in all_model_results.items():
                    m_info  = MODELS[m_name]
                    m_emoji = get_risk_emoji(m_res['category'])
                    active  = " ★" if m_name == selected_model else ""
                    rows_html += (
                        f'<tr>'
                        f'<td>{m_info["icon"]} {m_name}{active}</td>'
                        f'<td>—</td>'
                        f'<td>{m_emoji} {m_res["category"]}</td>'
                        f'<td>{m_res["confidence"]}%</td>'
                        f'</tr>'
                    )

                compare_html = (
                    '<div class="section-header">🔄 All Models Comparison</div>'
                    '<div class="holdings-container">'
                    '<table class="compare-table">'
                    '<thead><tr>'
                    '<th>Model</th>'
                    '<th>Math Score</th>'
                    '<th>AI Classification</th>'
                    '<th>AI Confidence</th>'
                    '</tr></thead>'
                    f'<tbody>{rows_html}</tbody>'
                    '</table>'
                    '</div>'
                    '<div class="section-divider"></div>'
                )
                st.markdown(compare_html, unsafe_allow_html=True)

            # ── Portfolio Metrics ────────────────────────────
            st.markdown('<div class="section-header">📊 Portfolio Metrics</div>', unsafe_allow_html=True)

            new_feats = results.get('new_features', {})

            def fmt(val, decimals=3, pct=False):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                return f"{val:.{decimals}f}{'%' if pct else ''}"

            st.markdown(
                '<div class="metrics-grid">'
                f'<div class="metric-card"><div class="metric-icon">📉</div><div class="metric-value">{vol:.2f}%</div><div class="metric-name">Volatility</div></div>'
                f'<div class="metric-card"><div class="metric-icon">⚖️</div><div class="metric-value">{beta:.2f}</div><div class="metric-name">Beta</div></div>'
                f'<div class="metric-card"><div class="metric-icon">🔀</div><div class="metric-value">{div_index:.2f}</div><div class="metric-name">Diversification</div></div>'
                f'<div class="metric-card"><div class="metric-icon">🏢</div><div class="metric-value">{port_cap_score:.2f}</div><div class="metric-name">Market Cap Score</div></div>'
                f'<div class="metric-card"><div class="metric-icon">📐</div><div class="metric-value">{fmt(new_feats.get("Portfolio_Downside_Volatility"))}</div><div class="metric-name">Downside Vol</div></div>'
                f'<div class="metric-card"><div class="metric-icon">📉</div><div class="metric-value">{fmt(new_feats.get("Portfolio_Max_Drawdown"))}</div><div class="metric-name">Max Drawdown</div></div>'
                f'<div class="metric-card"><div class="metric-icon">💧</div><div class="metric-value">{fmt(new_feats.get("Portfolio_Amihud_Illiquidity"))}</div><div class="metric-name">Illiquidity</div></div>'
                f'<div class="metric-card"><div class="metric-icon">🏦</div><div class="metric-value">{fmt(new_feats.get("Portfolio_Debt_to_Equity"))}</div><div class="metric-name">D/E Ratio</div></div>'
                f'<div class="metric-card"><div class="metric-icon">📈</div><div class="metric-value">{fmt(new_feats.get("Portfolio_Current_Ratio"))}</div><div class="metric-name">Current Ratio</div></div>'
                f'<div class="metric-card"><div class="metric-icon">🛡️</div><div class="metric-value">{fmt(new_feats.get("Portfolio_Interest_Coverage"))}</div><div class="metric-name">Int. Coverage</div></div>'
                f'<div class="metric-card"><div class="metric-icon">💰</div><div class="metric-value">{fmt(new_feats.get("Portfolio_ROA"))}</div><div class="metric-name">ROA</div></div>'
                f'<div class="metric-card"><div class="metric-icon">📊</div><div class="metric-value">{fmt(new_feats.get("Portfolio_Revenue_Growth_Vol"))}</div><div class="metric-name">Rev. Growth Vol</div></div>'
                '</div>',
                unsafe_allow_html=True
            )

            # ── Holdings Table ───────────────────────────────
            holdings_rows = ""
            for t, w in zip(tickers, weights):
                sector    = meta_df.loc[t, "Sector"] if (t in meta_df.index and "Sector" in meta_df.columns) else sector_map.get(t, "Unknown")
                cap_score = meta_df.loc[t, "Market_Cap_Score"] if t in meta_df.index else "N/A"
                bar_width = w * 100
                weight_bar = (
                    f'<div style="display:flex;align-items:center;gap:.5rem;">'
                    f'<div style="flex:1;height:6px;background:rgba(30,41,59,.8);border-radius:3px;overflow:hidden;">'
                    f'<div style="width:{bar_width}%;height:100%;background:linear-gradient(90deg,#3b82f6,#8b5cf6);border-radius:3px;"></div>'
                    f'</div>'
                    f'<span style="font-weight:600;min-width:45px;">{w*100:.1f}%</span>'
                    f'</div>'
                )
                holdings_rows += (
                    f'<tr>'
                    f'<td style="font-weight:600;color:#93c5fd;">{t}</td>'
                    f'<td>{sector}</td>'
                    f'<td>{weight_bar}</td>'
                    f'<td style="text-align:center;">{cap_score}</td>'
                    f'</tr>'
                )

            holdings_html = (
                '<div class="section-divider"></div>'
                '<div class="section-header">📋 Portfolio Holdings</div>'
                '<div class="holdings-container">'
                '<table class="sector-table">'
                '<thead><tr><th>Ticker</th><th>Sector</th><th>Weight</th><th>Cap Score</th></tr></thead>'
                f'<tbody>{holdings_rows}</tbody>'
                '</table>'
                '</div>'
            )
            st.markdown(holdings_html, unsafe_allow_html=True)

            # ── Sector Exposure ──────────────────────────────
            if portfolio_sectors:
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🏭 Sector Exposure</div>', unsafe_allow_html=True)

                colors = ["#3b82f6","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ef4444","#ec4899"]
                sector_cards_html = '<div class="sector-grid">'
                for idx, (sec, sec_w) in enumerate(
                    sorted(portfolio_sectors.items(), key=lambda x: x[1], reverse=True)
                ):
                    color = colors[idx % len(colors)]
                    sector_cards_html += (
                        f'<div class="sector-card" style="border-color:{color}40;">'
                        f'<div class="sector-pct" style="color:{color};">{sec_w*100:.1f}%</div>'
                        f'<div class="sector-name">{sec}</div>'
                        f'</div>'
                    )
                sector_cards_html += '</div>'
                st.markdown(sector_cards_html, unsafe_allow_html=True)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            error_msg = str(e).replace('<', '&lt;').replace('>', '&gt;')
            st.markdown(
                f'<div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);'
                f'border-radius:12px;padding:1.5rem;font-family:Inter,sans-serif;">'
                f'<div style="color:#f87171;font-weight:600;margin-bottom:.5rem;">&#10060; An Error Occurred</div>'
                f'<div style="color:rgba(200,210,230,.6);font-size:.9rem;">{error_msg}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; margin-top:2rem;">
        <div style="font-size:4rem; margin-bottom:1rem; opacity:.3;">🔍</div>
        <div style="font-family:'Inter',sans-serif;color:rgba(200,210,230,.4);font-size:1.2rem;font-weight:500;">
            Configure your portfolio above and click <strong>Analyze</strong>
        </div>
        <div style="font-family:'Inter',sans-serif;color:rgba(200,210,230,.25);font-size:.9rem;margin-top:.5rem;">
            Add stock tickers with their weights to get started
        </div>
    </div>

    <div style="display:flex; justify-content:center; gap:1.5rem; margin-top:3rem; flex-wrap:wrap;">
        <div style="background:rgba(30,41,59,.3);border:1px solid rgba(245,158,11,.15);border-radius:16px;padding:1.5rem 2rem;text-align:center;width:180px;">
            <div style="font-size:2rem; margin-bottom:.5rem;">🌲</div>
            <div style="font-family:'Inter',sans-serif;color:#f59e0b;font-size:.9rem;font-weight:600;">Random Forest</div>
            <div style="font-family:'Inter',sans-serif;color:rgba(200,210,230,.3);font-size:.75rem;margin-top:.3rem;">Rolling Window</div>
        </div>
        <div style="background:rgba(30,41,59,.3);border:1px solid rgba(167,139,250,.15);border-radius:16px;padding:1.5rem 2rem;text-align:center;width:180px;">
            <div style="font-size:2rem; margin-bottom:.5rem;">🧠</div>
            <div style="font-family:'Inter',sans-serif;color:#a78bfa;font-size:.9rem;font-weight:600;">LSTM</div>
            <div style="font-family:'Inter',sans-serif;color:rgba(200,210,230,.3);font-size:.75rem;margin-top:.3rem;">Rolling Window</div>
        </div>
        <div style="background:rgba(30,41,59,.3);border:1px solid rgba(52,211,153,.15);border-radius:16px;padding:1.5rem 2rem;text-align:center;width:180px;">
            <div style="font-size:2rem; margin-bottom:.5rem;">⚡</div>
            <div style="font-family:'Inter',sans-serif;color:#34d399;font-size:.9rem;font-weight:600;">Support Vector</div>
            <div style="font-family:'Inter',sans-serif;color:rgba(200,210,230,.3);font-size:.75rem;margin-top:.3rem;">Rolling Window</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
