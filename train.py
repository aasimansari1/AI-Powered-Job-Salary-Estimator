"""
Trains, evaluates, and persists five regression models.
Automatically generates synthetic data if the CSV does not exist.
Run:  python train.py
"""
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available – skipping.")

warnings.filterwarnings('ignore')


DATA_PATH = 'data/salary_data.csv'
MODEL_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
RESULTS_PATH = os.path.join(MODEL_DIR, 'model_results.json')


def ensure_data():
    if not os.path.exists(DATA_PATH):
        print("Data file not found. Generating synthetic dataset...")
        from generate_data import generate_salary_data
        os.makedirs('data', exist_ok=True)
        df = generate_salary_data(15000)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved {len(df)} rows to {DATA_PATH}")


def load_and_prepare():
    from preprocessing import clean_data, prepare_features, get_feature_names
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    df = clean_data(df)
    print(f"After cleaning: {len(df)} rows")
    X_arr, y = prepare_features(df)
    feature_names = get_feature_names()
    X = pd.DataFrame(X_arr, columns=feature_names)
    print(f"Feature matrix shape: {X.shape}")
    return X, y, df


def build_models():
    # Tuned for Streamlit Cloud free tier (1 GB RAM): reduced estimator counts
    models = {
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=120, max_depth=16, min_samples_leaf=4,
            n_jobs=-1, random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=250, learning_rate=0.06, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=250, learning_rate=0.06, max_depth=6,
            num_leaves=48, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbose=-1
        ),
    }
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostRegressor(
            iterations=250, learning_rate=0.06, depth=6,
            random_seed=42, verbose=0
        )
    return models


def evaluate(name, model, X_train, X_test, y_train, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_mean = cv_scores.mean()

    result = {
        'name': name,
        'r2': round(float(r2), 4),
        'mae': round(float(mae), 2),
        'rmse': round(float(rmse), 2),
        'cv_r2': round(float(cv_mean), 4),
        'train_time_s': round(train_time, 2),
    }

    print(
        f"  {name:<22} R²={r2:.4f}  MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  "
        f"CV-R²={cv_mean:.4f}  [{train_time:.1f}s]"
    )
    return result


def main():
    ensure_data()
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, _ = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}\n")

    models = build_models()
    results = []
    trained_models = {}

    print("Training and evaluating models:")
    print("-" * 75)
    for name, model in models.items():
        res = evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(res)
        trained_models[name] = model

    print("-" * 75)

    best = max(results, key=lambda r: r['r2'])
    best_model = trained_models[best['name']]

    print(f"\n✅  Best model: {best['name']}  (R²={best['r2']}, MAE=${best['mae']:,.0f})")

    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Saved best model → {BEST_MODEL_PATH}")

    with open(RESULTS_PATH, 'w') as f:
        json.dump({'results': results, 'best': best}, f, indent=2)
    print(f"Saved evaluation results → {RESULTS_PATH}")

    return best_model, results


if __name__ == '__main__':
    main()
