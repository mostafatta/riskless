import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ==========================================
# Feature Configuration
# ==========================================

# Original 6 portfolio-level features (keep as-is)
ORIGINAL_FEATURES = [
    'Portfolio_Volatility',
    'Portfolio_Beta',
    'Sector_Volatility',
    'Sector_Beta',
    'Diversification_Index',
    'Market_Cap_Score',
]

# New portfolio-level features derived from stock-level indicators
# via portfolio-weight aggregation (methodology §4–5)
NEW_FEATURES = [
    # ── Technical indicators (positively related to risk → higher = riskier) ──
    'Portfolio_Downside_Volatility',   # weighted avg of stock downside vol
    'Portfolio_Max_Drawdown',          # weighted avg of stock MDD
    'Portfolio_Amihud_Illiquidity',    # weighted avg of Amihud illiquidity

    # ── Financial indicators ──
    # Positively related to risk
    'Portfolio_Debt_to_Equity',        # weighted avg D/E ratio
    'Portfolio_Revenue_Growth_Vol',    # weighted avg revenue growth volatility
    # Negatively related to risk (higher = safer, normalized inverted)
    'Portfolio_Current_Ratio',         # weighted avg current ratio
    'Portfolio_Interest_Coverage',     # weighted avg ICR
    'Portfolio_ROA',                   # weighted avg return on assets
]

ALL_FEATURES = ORIGINAL_FEATURES + NEW_FEATURES


# ==========================================
# Min-Max Normalization (directional-aware)
# ==========================================

def normalize_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Apply per-column min-max normalization with directional alignment:
    - Positive-risk features → Z = (X - min) / (max - min)       [0=safe, 1=risky]
    - Negative-risk features → Z = 1 - (X - min) / (max - min)   [0=safe, 1=risky]

    Returns a new DataFrame with normalized columns; originals unchanged.
    """
    POSITIVE_RISK = {
        'Portfolio_Volatility', 'Portfolio_Beta',
        'Sector_Volatility', 'Sector_Beta',
        'Portfolio_Downside_Volatility', 'Portfolio_Max_Drawdown',
        'Portfolio_Amihud_Illiquidity', 'Portfolio_Debt_to_Equity',
        'Portfolio_Revenue_Growth_Vol',
    }
    NEGATIVE_RISK = {
        'Diversification_Index', 'Market_Cap_Score',
        'Portfolio_Current_Ratio', 'Portfolio_Interest_Coverage',
        'Portfolio_ROA',
    }

    df_out = df.copy()
    for col in features:
        col_min = df_out[col].min()
        col_max = df_out[col].max()
        rng = col_max - col_min
        if rng == 0:
            df_out[col] = 0.0
            continue
        if col in POSITIVE_RISK:
            df_out[col] = (df_out[col] - col_min) / rng
        elif col in NEGATIVE_RISK:
            df_out[col] = 1.0 - (df_out[col] - col_min) / rng
        # Unknown columns → leave raw (safe fallback)
    return df_out


# ==========================================
# Rolling Window Evaluation
# ==========================================

def rolling_window_evaluation(X, y, n_splits=5, test_size=0.15):
    """
    Rolling Window (Walk-Forward) Evaluation.
    Respects temporal order — no look-ahead bias.
    """
    n = len(X)
    test_len = int(n * test_size)
    total_train = n - (n_splits * test_len)

    if total_train <= 0:
        raise ValueError("n_splits is too large for the dataset size.")

    windows = []
    for i in range(n_splits):
        train_end  = total_train + i * test_len
        test_start = train_end
        test_end   = test_start + test_len
        train_idx  = list(range(0, train_end))
        test_idx   = list(range(test_start, test_end))
        windows.append((train_idx, test_idx))

    return windows


# ==========================================
# Main Training Function
# ==========================================

def train_model():
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data', 'processed', 'portfolio_dataset.csv'
    )

    if not os.path.exists(data_path):
        print("Error: Dataset not found! Run data_generator.py first.")
        return None

    df = pd.read_csv(data_path)

    # ── Feature availability check ──────────────────────────
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing_features   = [f for f in ALL_FEATURES if f not in df.columns]

    if missing_features:
        print(f"  [INFO] Missing new features — will use available subset:")
        for mf in missing_features:
            print(f"         • {mf}")
        print(f"  [INFO] Proceeding with {len(available_features)} features.\n")
    else:
        print(f"  [INFO] All {len(ALL_FEATURES)} features found.\n")

    features_to_use = available_features

    # ── Median imputation for any remaining NaNs ─────────────
    for col in features_to_use:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [IMPUTE] {col}: filled {n_nan} NaN(s) with median={median_val:.4f}")

    # ── Directional normalization ────────────────────────────
    df = normalize_features(df, features_to_use)

    X = df[features_to_use].values
    y = df['Risk_Category'].values

    print(f"Dataset loaded   : {len(df)} samples")
    print(f"Features used    : {len(features_to_use)}")
    print(f"Target classes   : {np.unique(y).tolist()}\n")

    print("=" * 55)
    print("  ROLLING WINDOW EVALUATION (Random Forest)")
    print("=" * 55)

    n_splits = 5
    windows  = rolling_window_evaluation(X, y, n_splits=n_splits, test_size=0.15)

    param_grid = [
        {'n_estimators': 100, 'max_depth':    5, 'min_samples_split':  2, 'min_samples_leaf': 1},
        {'n_estimators': 200, 'max_depth':   10, 'min_samples_split':  5, 'min_samples_leaf': 2},
        {'n_estimators': 300, 'max_depth':   15, 'min_samples_split': 10, 'min_samples_leaf': 4},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split':  2, 'min_samples_leaf': 1},
        {'n_estimators': 100, 'max_depth':   10, 'min_samples_split':  5, 'min_samples_leaf': 2},
    ]

    best_params  = None
    best_avg_acc = -1.0

    for params in param_grid:
        window_accuracies = []
        for train_idx, test_idx in windows:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            window_accuracies.append(accuracy_score(y_test, preds))

        avg_acc = np.mean(window_accuracies)
        print(f"  n_est={params['n_estimators']:3d} | depth={str(params['max_depth']):4s} "
              f"| split={params['min_samples_split']} | leaf={params['min_samples_leaf']} "
              f"→ Avg Accuracy: {avg_acc:.2%}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_params  = params

    print(f"\n  Best Params  : {best_params}")
    print(f"  Best Rolling Window Accuracy: {best_avg_acc:.2%}")

    # ==========================================
    # Final Model — Train on all except last 15%
    # ==========================================
    test_size_n    = int(len(X) * 0.15)
    X_train_final  = X[: -test_size_n]
    y_train_final  = y[: -test_size_n]
    X_test_final   = X[-test_size_n :]
    y_test_final   = y[-test_size_n :]

    best_model = RandomForestClassifier(random_state=42, **best_params)
    best_model.fit(X_train_final, y_train_final)

    test_preds = best_model.predict(X_test_final)
    test_acc   = accuracy_score(y_test_final, test_preds)

    print(f"\n{'='*55}")
    print(f"  TEST SET PERFORMANCE (FINAL — Last 15%)")
    print(f"{'='*55}")
    print(f"  Accuracy : {test_acc:.2%}\n")
    print(classification_report(y_test_final, test_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_final, test_preds))

    # ── Feature importance ──────────────────────────────────
    importances = best_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features_to_use, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False)
    print("\nFeature Importances:")
    print(fi_df.to_string(index=False))

    # ── Save artifacts ──────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "rf_rolling_window.pkl")
    joblib.dump(best_model, model_path)

    # Save the feature list so app.py can load it reliably
    feature_path = os.path.join(models_dir, "rf_features.pkl")
    joblib.dump(features_to_use, feature_path)

    print(f"\nModel saved to   : {model_path}")
    print(f"Features saved to: {feature_path}")

    return best_model


if __name__ == "__main__":
    train_model()
