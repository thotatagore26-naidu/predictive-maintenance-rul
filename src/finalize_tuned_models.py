"""
Assemble the final tuned models using the best hyperparameters found
by Optuna for each, retrain on train+val combined, and evaluate
honestly on the held-out test set.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

RANDOM_STATE = 42

# Best params found during tuning (XGBoost/LightGBM from the first run,
# CatBoost from the separate trimmed-search run)
XGB_PARAMS = {'n_estimators': 425, 'max_depth': 3, 'learning_rate': 0.03557869786001109,
              'subsample': 0.7840560190635382, 'colsample_bytree': 0.8933131797927751,
              'min_child_weight': 2}
LGB_PARAMS = {'n_estimators': 207, 'max_depth': 10, 'learning_rate': 0.022007421575163263,
              'num_leaves': 37, 'subsample': 0.9027535926620921,
              'colsample_bytree': 0.6033281493105744, 'min_child_samples': 36}

with open('data/catboost_params.pkl', 'rb') as f:
    CB_PARAMS = pickle.load(f)


def phm08_score(y_true, y_pred):
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(score)


def main():
    train = pd.read_csv('data/train_features.csv')
    test = pd.read_csv('data/test_features.csv')
    with open('data/feature_cols.txt') as f:
        feature_cols = f.read().splitlines()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(train, groups=train['unit_nr']))
    tr, val = train.iloc[train_idx], train.iloc[val_idx]

    X_full = pd.concat([tr[feature_cols], val[feature_cols]])
    y_full = pd.concat([tr['RUL'], val['RUL']])
    X_test, y_test = test[feature_cols], test['RUL']

    models = {
        'XGBoost': xgb.XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        'CatBoost': cb.CatBoostRegressor(**CB_PARAMS, random_state=RANDOM_STATE, verbose=0),
    }

    print(f"{'Model':<12} {'Test RMSE':>11} {'Test PHM08':>12}")
    print("-" * 37)

    test_preds = {}
    for name, model in models.items():
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
            'params': {'xgb': XGB_PARAMS, 'lgb': LGB_PARAMS, 'cb': CB_PARAMS},
            'models': models,
        }, f)

    print("\nSaved tuned models and predictions to data/tuned_results.pkl")


if __name__ == "__main__":
    main()
