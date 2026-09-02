"""
Baseline models: one engine-level train/val split, three models with
default hyperparameters, evaluated on both RMSE and the field's actual
scoring metric (PHM08 asymmetric score).

PHM08 SCORE EXPLAINED:
Standard RMSE treats an early prediction and a late prediction as
equally bad if the error magnitude is the same. But in maintenance,
predicting RUL too LATE (telling an engineer "you have 50 cycles left"
when really there are only 10) is far more dangerous than predicting
too EARLY (being overly cautious). The PHM08 competition's official
scoring function reflects this with an asymmetric exponential penalty:
small penalty for early predictions, much larger penalty for late ones.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

RANDOM_STATE = 42


def phm08_score(y_true, y_pred):
    """Official PHM08 asymmetric scoring function.
    Penalizes late predictions (underestimating RUL) much more heavily
    than early predictions (overestimating RUL)."""
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(score)


def main():
    train = pd.read_csv('data/train_features.csv')
    test = pd.read_csv('data/test_features.csv')
    with open('data/feature_cols.txt') as f:
        feature_cols = f.read().splitlines()

    # Engine-level split: 80 engines train, 20 engines validation
    # (critical: split by unit_nr, never by row, to avoid leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(train, groups=train['unit_nr']))
    tr, val = train.iloc[train_idx], train.iloc[val_idx]

    print(f"Train: {tr['unit_nr'].nunique()} engines, {len(tr)} rows")
    print(f"Val:   {val['unit_nr'].nunique()} engines, {len(val)} rows")

    X_tr, y_tr = tr[feature_cols], tr['RUL']
    X_val, y_val = val[feature_cols], val['RUL']
    X_test, y_test = test[feature_cols], test['RUL']

    models = {
        'XGBoost': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        'CatBoost': cb.CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
    }

    print(f"\n{'Model':<12} {'Val RMSE':>10} {'Val PHM08':>12} {'Test RMSE':>11} {'Test PHM08':>12}")
    print("-" * 60)

    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_phm08 = phm08_score(y_val.values, val_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_phm08 = phm08_score(y_test.values, test_pred)

        print(f"{name:<12} {val_rmse:>10.2f} {val_phm08:>12.1f} {test_rmse:>11.2f} {test_phm08:>12.1f}")
        results[name] = {'model': model, 'test_pred': test_pred, 'test_rmse': test_rmse}

    # Save for later use (ensemble stage)
    import pickle
    with open('data/baseline_results.pkl', 'wb') as f:
        pickle.dump({'y_test': y_test.values, 'results': {k: v['test_pred'] for k, v in results.items()}}, f)

    print("\nSaved baseline predictions for ensemble stage.")


if __name__ == "__main__":
    main()
