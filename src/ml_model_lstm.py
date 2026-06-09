import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

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
# LSTM Architecture
# ==========================================

def build_lstm_model(input_shape, num_classes, units=64, dropout=0.2, lr=1e-3):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=False),
        Dense(32, activation='relu'),
        Dropout(dropout),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


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

    le          = LabelEncoder()
    y_raw       = le.fit_transform(df['Risk_Category'].values)
    num_classes = len(le.classes_)

    # StandardScaler after directional normalization
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset loaded   : {len(df)} samples")
    print(f"Features used    : {len(features_to_use)}")
    print(f"Target classes   : {le.classes_.tolist()}\n")

    print("=" * 60)
    print("  ROLLING WINDOW EVALUATION (LSTM)")
    print("=" * 60)

    input_shape = (1, len(features_to_use))
    n_splits    = 5
    windows     = rolling_window_evaluation(len(X_scaled), n_splits=n_splits, test_size=0.15)

    param_grid = [
        {'units': 32, 'dropout': 0.1, 'epochs': 80,  'batch_size': 16, 'lr': 1e-3},
        {'units': 64, 'dropout': 0.2, 'epochs': 100, 'batch_size': 8,  'lr': 1e-3},
    ]

    best_params  = None
    best_avg_acc = -1.0

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)

    for params in param_grid:
        window_accuracies = []

        for train_idx, test_idx in windows:
            X_train_raw = X_scaled[train_idx]
            y_train_raw = y_raw[train_idx]
            X_test_raw  = X_scaled[test_idx]
            y_test      = y_raw[test_idx]

            class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train_raw), y=y_train_raw)
            class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

            X_train_3d  = X_train_raw.reshape(X_train_raw.shape[0], 1, X_train_raw.shape[1])
            X_test_3d   = X_test_raw.reshape(X_test_raw.shape[0], 1, X_test_raw.shape[1])
            y_train_cat = to_categorical(y_train_raw, num_classes)

            model = build_lstm_model(
                (1, len(features_to_use)), num_classes,
                units=params['units'], dropout=params['dropout'], lr=params['lr']
            )
            model.fit(
                X_train_3d, y_train_cat,
                epochs=params['epochs'], batch_size=params['batch_size'],
                validation_split=0.1, callbacks=[early_stop, reduce_lr],
                class_weight=class_weight_dict, verbose=0
            )

            preds_proba = model.predict(X_test_3d, verbose=0)
            y_pred      = np.argmax(preds_proba, axis=1)
            window_accuracies.append(accuracy_score(y_test, y_pred))

        avg_acc = np.mean(window_accuracies)
        print(f"  units={params['units']:3d} | drop={params['dropout']} | ep={params['epochs']:3d} "
              f"| batch={params['batch_size']:2d} | lr={params['lr']:.0e}  →  Avg Accuracy: {avg_acc:.2%}")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_params  = params

    print(f"\n  Best Params  : {best_params}")
    print(f"  Best Rolling Window Accuracy : {best_avg_acc:.2%}")

    # ==========================================
    # Final Model
    # ==========================================
    test_size_n  = int(len(X_scaled) * 0.15)
    X_train_raw  = X_scaled[: -test_size_n]
    y_train_raw  = y_raw[: -test_size_n]
    X_test_raw   = X_scaled[-test_size_n :]
    y_test_final = y_raw[-test_size_n :]

    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train_raw), y=y_train_raw)
    class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

    X_train_final = X_train_raw.reshape(X_train_raw.shape[0], 1, X_train_raw.shape[1])
    X_test_final  = X_test_raw.reshape(X_test_raw.shape[0], 1, X_test_raw.shape[1])
    y_train_final = to_categorical(y_train_raw, num_classes)

    final_model = build_lstm_model(
        (1, len(features_to_use)), num_classes,
        units=best_params['units'], dropout=best_params['dropout'], lr=best_params['lr']
    )

    print("\nTraining Final Model...")
    final_model.fit(
        X_train_final, y_train_final,
        epochs=best_params['epochs'], batch_size=best_params['batch_size'],
        validation_split=0.1, callbacks=[early_stop, reduce_lr],
        class_weight=class_weight_dict, verbose=1
    )

    preds_proba  = final_model.predict(X_test_final, verbose=0)
    y_pred_final = np.argmax(preds_proba, axis=1)
    test_acc     = accuracy_score(y_test_final, y_pred_final)

    print(f"\n{'='*60}")
    print(f"  TEST SET PERFORMANCE (FINAL — Last 15%)")
    print(f"{'='*60}")
    print(f"  Accuracy: {test_acc:.2%}\n")
    print(classification_report(y_test_final, y_pred_final, target_names=le.classes_, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_final, y_pred_final))

    # ── Save artifacts ──────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    final_model.save(os.path.join(models_dir, "lstm_rolling_window.keras"))
    joblib.dump(scaler,          os.path.join(models_dir, "lstm_scaler.pkl"))
    joblib.dump(le,              os.path.join(models_dir, "lstm_label_encoder.pkl"))
    joblib.dump(features_to_use, os.path.join(models_dir, "lstm_features.pkl"))

    print(f"\nModel saved to   : {models_dir}/lstm_rolling_window.keras")
    print(f"Features saved to: {models_dir}/lstm_features.pkl")

    return final_model


if __name__ == "__main__":
    train_model()
