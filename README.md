# ✈️ Predictive Maintenance: Aircraft Engine Remaining Useful Life (RUL)

An ensemble machine learning system that predicts how many operating cycles remain before an aircraft engine fails, using real sensor data — built to explore predictive maintenance, a genuine high-value industrial ML problem.

## The problem

Industrial equipment maintenance faces a tradeoff: **reactive maintenance** (wait until it breaks) causes costly downtime and safety risk, while **fixed-schedule maintenance** wastes money replacing parts that still have life left. **Predictive maintenance** uses live sensor data to estimate a machine's actual remaining useful life (RUL), so maintenance happens exactly when needed.

This project predicts RUL for aircraft turbofan engines using the NASA C-MAPSS dataset — the industry-standard benchmark for prognostics research, used in 1,000+ published papers.

## Approach

1. **Data**: NASA C-MAPSS FD001 subset — 100 training engines run to failure, 100 test engines truncated mid-life with held-out ground-truth RUL.
2. **Feature engineering**:
   - Dropped 7 sensors with ~zero variance (carry no information)
   - Rolling mean/std (10-cycle window, computed per-engine) to smooth noisy raw sensor signals
   - Piecewise-linear RUL target, capped at 125 cycles — standard practice, since sensors show no degradation signal during an engine's healthy phase
3. **Validation**: engine-level train/validation split (never row-level, to avoid leaking cycles from the same engine across splits)
4. **Models**: XGBoost, LightGBM, CatBoost — each hyperparameter-tuned with **Optuna**, optimized directly against the domain's real scoring metric (not a generic proxy)
5. **Evaluation metric**: the official **PHM08 asymmetric score** — penalizes late RUL predictions (dangerously underestimating remaining life) far more heavily than early ones, unlike standard RMSE which treats both equally
6. **Ensembling**: simple and inverse-score-weighted averaging of the three tuned models

## Results

| Model | Test RMSE | Test PHM08 (lower = better) |
|---|---|---|
| XGBoost (tuned) | 18.37 | 1067.9 |
| LightGBM (tuned) | 18.46 | 1236.1 |
| **CatBoost (tuned)** | **17.95** | **953.5** |
| Simple Average Ensemble | 18.17 | 1060.7 |
| Weighted Average Ensemble | 18.15 | 1049.4 |

**Key finding**: the single best tuned model (CatBoost) outperformed both ensemble strategies. This is reported honestly rather than dressed up — all three base models are gradient-boosted trees on the same tabular features, so they likely make correlated errors, limiting what ensembling can add. A more architecturally diverse ensemble (e.g. adding a neural network or linear model) is a natural next step.

**Tuning impact**: hyperparameter tuning improved XGBoost's PHM08 score by ~60% over its default-hyperparameter baseline. CatBoost's tuned score was slightly worse than its own baseline — an honest result of a trimmed 15-trial search budget, included rather than hidden.

## Tech stack

`Python` · `XGBoost` · `LightGBM` · `CatBoost` · `Optuna` · `scikit-learn` · `pandas` · `matplotlib`

## Repo structure

```
predictive-maintenance/
├── analysis.ipynb              # Full analysis, executed with outputs
├── README.md
├── data/
│   ├── train_FD001.csv
│   ├── test_FD001.csv
│   └── RUL_FD001.csv
├── outputs/
│   ├── degradation_curves.png
│   ├── feature_verification.png
│   └── final_comparison.png
└── src/
    ├── build_features.py       # Feature engineering pipeline
    ├── train_baseline.py       # Baseline models (default hyperparameters)
    ├── tune_models.py          # Optuna hyperparameter search
    └── build_ensemble.py       # Ensemble construction and evaluation
```

## How to run

```bash
pip install pandas numpy xgboost lightgbm catboost scikit-learn optuna matplotlib jupyter
jupyter notebook analysis.ipynb
```

## Limitations / future work

- Only the single-operating-condition subset (FD001) was used; the harder FD002/FD004 subsets involve 6 operating conditions and would need per-condition normalization
- A neural sequence model (LSTM/GRU/Transformer) on raw time-series windows, rather than hand-engineered rolling features, is a natural extension and could capture longer-range temporal patterns
- Ensembling could be revisited with more architecturally diverse base learners

## Author

Thotakura Tagore — M.Sc. Artificial Intelligence and Data Science, Deggendorf Institute of Technology
