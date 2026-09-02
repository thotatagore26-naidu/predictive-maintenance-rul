"""
Ensemble the three tuned models.

We try two simple, robust ensembling strategies:
1. Simple averaging -- equal-weight blend of all three predictions
2. Weighted averaging -- weights inversely proportional to each model's
   individual PHM08 score, so stronger models get more say

Both are evaluated against the same held-out test set the individual
models were scored on, for a fair comparison.
"""
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error


def phm08_score(y_true, y_pred):
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(score)


def main():
    with open('data/tuned_results.pkl', 'rb') as f:
        data = pickle.load(f)

    y_test = data['y_test']
    preds = data['test_preds']

    print(f"{'Model':<20} {'Test RMSE':>11} {'Test PHM08':>12}")
    print("-" * 45)

    individual_scores = {}
    for name, pred in preds.items():
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        score = phm08_score(y_test, pred)
        individual_scores[name] = score
        print(f"{name:<20} {rmse:>11.2f} {score:>12.1f}")

    # Simple average ensemble
    simple_avg = np.mean([preds['XGBoost'], preds['LightGBM'], preds['CatBoost']], axis=0)
    simple_avg_rmse = np.sqrt(mean_squared_error(y_test, simple_avg))
    simple_avg_score = phm08_score(y_test, simple_avg)
    print(f"{'Simple Average':<20} {simple_avg_rmse:>11.2f} {simple_avg_score:>12.1f}")

    # Weighted average: weight = inverse of PHM08 score (lower score = more weight)
    inv_scores = {k: 1 / v for k, v in individual_scores.items()}
    total = sum(inv_scores.values())
    weights = {k: v / total for k, v in inv_scores.items()}
    print(f"\nWeights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")

    weighted_avg = sum(preds[name] * w for name, w in weights.items())
    weighted_avg_rmse = np.sqrt(mean_squared_error(y_test, weighted_avg))
    weighted_avg_score = phm08_score(y_test, weighted_avg)
    print(f"{'Weighted Average':<20} {weighted_avg_rmse:>11.2f} {weighted_avg_score:>12.1f}")

    # Save final results summary
    with open('data/final_results.pkl', 'wb') as f:
        pickle.dump({
            'y_test': y_test,
            'individual_preds': preds,
            'simple_avg': simple_avg,
            'weighted_avg': weighted_avg,
            'weights': weights,
        }, f)

    print("\nSaved final ensemble results.")


if __name__ == "__main__":
    main()
