"""
Hyperparameter tuning with Optuna.

We use the PHM08 score (not RMSE) as the optimization objective, since
that's the metric that actually reflects what matters in this domain
(penalizing dangerous late predictions much more than early ones).
Each model gets its own study since their hyperparameter spaces differ.

We tune against the validation set (engine-level split, same as
baseline) and keep the held-out test set completely untouched until
final evaluation, to get an honest read of generalization.
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pickle
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_TRIALS = 40


def phm08_score(y_true, y_pred):
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(score)


def load_data():
    train = pd.read_csv('data/train_features.csv')
    test = pd.read_csv('data/test_features.csv')
    with open('data/feature_cols.txt') as f:
        feature_cols = f.read().splitlines()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(train, groups=train['unit_nr']))
    tr, val = train.iloc[train_idx], train.iloc[val_idx]

    return (tr[feature_cols], tr['RUL'], val[feature_cols], val['RUL'],
            test[feature_cols], test['RUL'], feature_cols)


def tune_xgboost(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return phm08_score(y_val.values, pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def tune_lightgbm(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return phm08_score(y_val.values, pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def tune_catboost(X_tr, y_tr, X_val, y_val):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 600),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_state': RANDOM_STATE,
            'verbose': 0,
        }
        model = cb.CatBoostRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return phm08_score(y_val.values, pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def main():
    X_tr, y_tr, X_val, y_val, X_test, y_test, feature_cols = load_data()

    print(f"Tuning with {N_TRIALS} trials per model (this will take a few minutes)...\n")

    print("Tuning XGBoost...")
    xgb_params = tune_xgboost(X_tr, y_tr, X_val, y_val)
    print(f"  Best params: {xgb_params}")

    print("\nTuning LightGBM...")
    lgb_params = tune_lightgbm(X_tr, y_tr, X_val, y_val)
    print(f"  Best params: {lgb_params}")

    print("\nTuning CatBoost...")
    cb_params = tune_catboost(X_tr, y_tr, X_val, y_val)
    print(f"  Best params: {cb_params}")

    # Retrain final tuned models on train+val combined for max data usage,
    # then evaluate honestly on the held-out test set
    X_full = pd.concat([X_tr, X_val])
    y_full = pd.concat([y_tr, y_val])

    final_models = {
        'XGBoost': xgb.XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(**lgb_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        'CatBoost': cb.CatBoostRegressor(**cb_params, random_state=RANDOM_STATE, verbose=0),
    }

    from sklearn.metrics import mean_squared_error
    print(f"\n{'Model':<12} {'Test RMSE':>11} {'Test PHM08':>12}")
    print("-" * 37)

    test_preds = {}
    for name, model in final_models.items():
        model.fit(X_full, y_full)
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        score = phm08_score(y_test.values, pred)
        print(f"{name:<12} {rmse:>11.2f} {score:>12.1f}")
        test_preds[name] = pred

    with open('data/tuned_results.pkl', 'wb') as f:
        pickle.dump({
            'y_test': y_test.values,
            'test_preds': test_preds,
            'params': {'xgb': xgb_params, 'lgb': lgb_params, 'cb': cb_params},
            'models': final_models,
        }, f)

    print("\nSaved tuned models and predictions.")


if __name__ == "__main__":
    main()
