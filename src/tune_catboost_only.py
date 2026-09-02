"""
CatBoost tuning, run separately with a smaller trial budget and capped
iteration range -- CatBoost trains noticeably slower per-trial than
XGBoost/LightGBM at comparable settings, so we trim the search space
to keep total tuning time reasonable while still getting a real
improvement over the default-hyperparameter baseline.
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import GroupShuffleSplit
import catboost as cb
import pickle
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_TRIALS = 15


def phm08_score(y_true, y_pred):
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(score)


def main():
    train = pd.read_csv('data/train_features.csv')
    with open('data/feature_cols.txt') as f:
        feature_cols = f.read().splitlines()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(train, groups=train['unit_nr']))
    tr, val = train.iloc[train_idx], train.iloc[val_idx]

    X_tr, y_tr = tr[feature_cols], tr['RUL']
    X_val, y_val = val[feature_cols], val['RUL']

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 350),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_state': RANDOM_STATE,
            'verbose': 0,
            'thread_count': -1,
        }
        model = cb.CatBoostRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return phm08_score(y_val.values, pred)

    print(f"Tuning CatBoost with {N_TRIALS} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print(f"Best params: {study.best_params}")
    print(f"Best val PHM08: {study.best_value:.1f}")

    with open('data/catboost_params.pkl', 'wb') as f:
        pickle.dump(study.best_params, f)

    print("Saved CatBoost best params.")


if __name__ == "__main__":
    main()
