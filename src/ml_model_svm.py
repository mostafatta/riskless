import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    - Positive-risk features → Z = (X - min) / (max - min)
    - Negative-risk features → Z = 1 - (X - min) / (max - min)
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
    return df_out


# ==========================================
# Rolling Window Evaluation
# ==========================================

def rolling_window_evaluation(n_samples, n_splits=5, test_size=0.15):
    """
    Rolling Window (Walk-Forward) Evaluation.
    Respects temporal order — no look-ahead bias.
    """
    test_len    = int(n_samples * test_size)
    total_train = n_samples - (n_splits * test_len)

    if total_train <= 0:
        raise ValueError("n_splits is too large for the dataset size.")

    windows = []
    for i in range(n_splits):
        train_end  = total_train + i * test_len
        test_start = train_end
        test_end   = test_start + test_len
        windows.append((list(range(0, train_end)), list(range(test_start, test_end))))

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

    le = LabelEncoder()
    y  = le.fit_transform(df['Risk_Category'].values)

    scaler = StandardScaler()
    X      = scaler.fit_transform(df[features_to_use].values)

    print(f"Dataset loaded   : {len(df)} samples")
    print(f"Features used    : {len(features_to_use)}")
    print(f"Target classes   : {le.classes_.tolist()}\n")

    print("=" * 55)
    print("  ROLLING WINDOW EVALUATION (SVM)")
    print("=" * 55)

    n_splits = 5
    windows  = rolling_window_evaluation(len(X), n_splits=n_splits, test_size=0.15)

    param_grid = [
        {'C':  0.1, 'kernel': 'rbf',    'gamma': 'scale'},
        {'C':  1.0, 'kernel': 'rbf',    'gamma': 'scale'},
        {'C': 10.0, 'kernel': 'rbf',    'gamma': 'scale'},
        {'C':  1.0, 'kernel': 'linear', 'gamma': 'scale'},
        {'C': 10.0, 'kernel': 'linear', 'gamma': 'scale'},
    ]

    best_params  = None
    best_avg_acc = -1.0

    for params in param_grid:
        window_accuracies = []

        for train_idx, test_idx in windows:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = SVC(
                C=params['C'], kernel=params['kernel'],
                gamma=params['gamma'], random_state=42, probability=True
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            window_accuracies.append(accuracy_score(y_test, preds))

        avg_acc = np.mean(window_accuracies)
        print(f"  C={params['C']:5.1f} | kernel={params['kernel']:6s} | gamma={params['gamma']} "
              f"→ Avg Accuracy: {avg_acc:.2%}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_params  = params

    print(f"\n  Best Params  : {best_params}")
    print(f"  Best Rolling Window Accuracy: {best_avg_acc:.2%}")

    # ==========================================
    # Final Model
    # ==========================================
    test_size_n   = int(len(X) * 0.15)
    X_train_final = X[: -test_size_n]
    y_train_final = y[: -test_size_n]
    X_test_final  = X[-test_size_n :]
    y_test_final  = y[-test_size_n :]

    best_model = SVC(
        C=best_params['C'], kernel=best_params['kernel'],
        gamma=best_params['gamma'], random_state=42, probability=True
    )
    best_model.fit(X_train_final, y_train_final)

    y_pred   = best_model.predict(X_test_final)
    test_acc = accuracy_score(y_test_final, y_pred)

    print(f"\n{'='*55}")
    print(f"  TEST SET PERFORMANCE (FINAL — Last 15%)")
    print(f"{'='*55}")
    print(f"  Accuracy: {test_acc:.2%}\n")
    print(classification_report(y_test_final, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_final, y_pred))

    # ── Save artifacts ──────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(best_model,      os.path.join(models_dir, "svm_rolling_window.pkl"))
    joblib.dump(scaler,          os.path.join(models_dir, "svm_scaler.pkl"))
    joblib.dump(le,              os.path.join(models_dir, "svm_label_encoder.pkl"))
    joblib.dump(features_to_use, os.path.join(models_dir, "svm_features.pkl"))

    print(f"\nModel saved to   : {models_dir}/svm_rolling_window.pkl")
    print(f"Features saved to: {models_dir}/svm_features.pkl")

    return best_model


if __name__ == "__main__":
    train_model()
