"""
Feature engineering for the RUL prediction task.

Two key domain-specific decisions here, both standard practice in the
prognostics literature (Saxena et al. 2008 and follow-up work):

1. RUL CAPPING: in the healthy phase of an engine's life, sensors show
   essentially no signal of degradation (see degradation_curves.png --
   flat and noisy for the first ~100+ cycles). If we ask the model to
   predict a perfectly linear decreasing RUL from cycle 1 onward, we're
   penalizing it for failing to predict something that genuinely isn't
   observable in the data yet. The standard fix is a piecewise-linear
   RUL target: cap it at some max value (commonly 125), so the model
   only has to learn "this engine is healthy" during the flat period,
   and focuses its learning capacity on the degradation phase where the
   signal actually exists.

2. ROLLING WINDOW FEATURES: single-cycle sensor readings are noisy
   (visible in the plots). Rolling mean/std over a window of recent
   cycles smooths that noise out and lets the model see the underlying
   trend, which is what actually correlates with remaining life.
"""
import pandas as pd
import numpy as np

RUL_CAP = 125
ROLLING_WINDOW = 10

# Sensors with ~zero variance in FD001 -- carry no information, drop them
CONSTANT_SENSORS = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
CONSTANT_SETTINGS = ['setting_3']

DROP_COLS = CONSTANT_SENSORS + CONSTANT_SETTINGS


def add_rolling_features(df: pd.DataFrame, sensor_cols: list[str], window: int) -> pd.DataFrame:
    """Add rolling mean and std for each sensor, computed per-engine so
    windows never cross between different engines' histories."""
    df = df.sort_values(['unit_nr', 'time_cycles']).copy()

    grouped = df.groupby('unit_nr')[sensor_cols]
    rolling_mean = grouped.rolling(window=window, min_periods=1).mean().reset_index(drop=True)
    rolling_std = grouped.rolling(window=window, min_periods=1).std().fillna(0).reset_index(drop=True)

    rolling_mean.columns = [f"{c}_rollmean{window}" for c in sensor_cols]
    rolling_std.columns = [f"{c}_rollstd{window}" for c in sensor_cols]

    df = pd.concat([df.reset_index(drop=True), rolling_mean, rolling_std], axis=1)
    return df


def prepare_train(train: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    train = train.drop(columns=DROP_COLS)
    sensor_cols = [c for c in train.columns if c.startswith('s_')]

    # Apply the RUL cap (piecewise-linear target)
    train['RUL'] = train['RUL'].clip(upper=RUL_CAP)

    train = add_rolling_features(train, sensor_cols, ROLLING_WINDOW)

    feature_cols = [c for c in train.columns
                    if c not in ['unit_nr', 'time_cycles', 'RUL']]
    return train, feature_cols


def prepare_test(test: pd.DataFrame, rul_ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Test engines are truncated mid-life. We take the LAST row of each
    engine's history (its current state) and predict from there -- that's
    what the ground-truth RUL file corresponds to."""
    test = test.drop(columns=DROP_COLS)
    sensor_cols = [c for c in test.columns if c.startswith('s_')]

    test = add_rolling_features(test, sensor_cols, ROLLING_WINDOW)

    last_cycle = test.groupby('unit_nr').tail(1).reset_index(drop=True)
    last_cycle['RUL'] = rul_ground_truth['RUL'].values
    last_cycle['RUL'] = last_cycle['RUL'].clip(upper=RUL_CAP)

    return last_cycle


def main():
    train = pd.read_csv('data/train_FD001.csv')
    test = pd.read_csv('data/test_FD001.csv')
    rul = pd.read_csv('data/RUL_FD001.csv')

    print("Building training features...")
    train_processed, feature_cols = prepare_train(train)
    print(f"  {len(feature_cols)} features, {len(train_processed)} rows "
          f"({train_processed['unit_nr'].nunique()} engines)")

    print("Building test features (last cycle per engine only)...")
    test_processed = prepare_test(test, rul)
    print(f"  {len(test_processed)} rows (one per test engine)")

    train_processed.to_csv('data/train_features.csv', index=False)
    test_processed.to_csv('data/test_features.csv', index=False)

    with open('data/feature_cols.txt', 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"\nSaved train_features.csv, test_features.csv, feature_cols.txt")
    print(f"\nRUL distribution after capping at {RUL_CAP}:")
    print(train_processed['RUL'].describe())


if __name__ == "__main__":
    main()
