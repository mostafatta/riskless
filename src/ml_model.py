import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib


def train_model():
    # ==========================================
    # 1. Load Dataset
    # ==========================================
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data', 'processed', 'portfolio_dataset.csv'
    )

    if not os.path.exists(data_path):
        print("Error: Dataset not found!")
        print("Run data_generator.py first.")
        return None

    df = pd.read_csv(data_path)

    # ==========================================
    # 2. Validate Required Columns
    # ==========================================
    expected_features = [
        'Portfolio_Volatility',
        'Portfolio_Beta',
        'Sector_Volatility',
        'Sector_Beta',
        'Diversification_Index',
        'Market_Cap_Score',
    ]

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print("Regenerate dataset with data_generator.py")
        return None

    X = df[expected_features]
    y = df['Risk_Category']

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Features: {expected_features}")
    print(f"Target classes: {y.unique().tolist()}\n")

    # ==========================================
    # 3. Split: 70% Train, 15% Validation, 15% Test
    # ==========================================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ==========================================
    # 4. Hyperparameter Tuning with GridSearchCV
    # ==========================================
    print("\nRunning Hyperparameter Tuning (GridSearchCV)...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.2%}")

    # ==========================================
    # 5. Evaluate on Validation Set
    # ==========================================
    val_predictions = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"\n{'='*40}")
    print(f"  VALIDATION SET PERFORMANCE")
    print(f"{'='*40}")
    print(f"  Accuracy: {val_accuracy:.2%}")
    print(f"\n{classification_report(y_val, val_predictions)}")

    # ==========================================
    # 6. Evaluate on Test Set (Final)
    # ==========================================
    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"{'='*40}")
    print(f"  TEST SET PERFORMANCE (FINAL)")
    print(f"{'='*40}")
    print(f"  Accuracy: {test_accuracy:.2%}")
    print(f"\n{classification_report(y_test, test_predictions)}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # ==========================================
    # 7. Feature Importance
    # ==========================================
    print(f"\n{'='*40}")
    print("  FEATURE IMPORTANCE")
    print(f"{'='*40}")

    importances = best_model.feature_importances_
    for feat, imp in sorted(
        zip(expected_features, importances),
        key=lambda x: x[1],
        reverse=True,
    ):
        bar = "█" * int(imp * 50)
        print(f"  {feat:30s} {imp:.4f} {bar}")

    # ==========================================
    # 8. Save Model
    # ==========================================
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'models'
    )
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "risk_classifier.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")

    return best_model


if __name__ == "__main__":
    train_model()