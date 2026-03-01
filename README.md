# IPL Run Chase Score Prediction via Gradient Boosted Regression

This repository presents an investigation into the predictability of second-innings run totals in Indian Premier League (IPL) T20 cricket, using ball-by-ball match data spanning the 2008–2025 seasons. An XGBoost regression model is trained to estimate the final innings score from a set of derived in-match features, with a focus on quantifying prediction uncertainty through an empirical predictive distribution.

---

## Performance Summary

Model selection was conducted by cross-validated mean absolute error (MAE) across four candidate regressors. The final tuned XGBoost model achieves a 46% reduction in MAE over the linear baseline.

| Model | MAE (runs) | RMSE (runs) | MAPE | R² |
|-------|-----------|------------|------|----|
| **XGBoost (tuned)** | **11.35** | **17.32** | **7.77%** | **0.6454** |

A further validation finding of note: conditional mean absolute error scales approximately linearly with balls remaining (slope ≈ 0.98 runs/ball, R ≈ 0.98), consistent with the intuition that outcome spread decreases as the innings progresses.

---

## Data & Experimental Design

**Source:** Ball-by-ball IPL delivery data (2008–2025), retrieved from [Kaggle](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025). The dataset is filtered to second-innings chase scenarios exclusively; first-innings records are excluded.

**Train/test partitioning:** A season-based split is employed to prevent temporal data leakage. Test seasons (2019, 2022, 2023) were held out entirely from model fitting and hyperparameter search. This partitioning strategy reflects realistic deployment conditions, where the model is evaluated on seasons unseen during training.

---

## Feature Engineering

Raw delivery-level data is aggregated to produce five model features, selected via permutation importance analysis on a held-out validation set. Features encoding runs target directly were found to introduce a ceiling bias and were excluded.

| Feature | Derivation |
|---------|-----------|
| `current_rr` | `team_runs / (balls_faced / 6)` |
| `required_rr` | `(target − team_runs) / (balls_left / 6)` |
| `balls_left` | `120 − balls_faced` |
| `wickets` | `10 − team_wicket` |
| `batting_team` | Target-encoded categorical |

Permutation importance results indicated that `strike_rate` and `economy` (mapped from career-average batter/bowler statistics) contributed negligible predictive value and were dropped from the final feature set.

---

## Modelling Pipeline

Preprocessing is implemented as a `scikit-learn` `ColumnTransformer`:

- `current_rr` — `StandardScaler`
- `required_rr` — `RobustScaler`
- `balls_left`, `wickets` — `MinMaxScaler`
- `batting_team` — `TargetEncoder` (category-encoders)

The preprocessor is composed with an `XGBRegressor` in a `Pipeline` object and tuned via `RandomizedSearchCV` over the parameter space `{n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight}`.

---

## Predictive Distribution Construction

Point estimates from the regression model are augmented with an empirical predictive distribution. Given a new observation with `balls_left = b`, the set of historical residuals from the test set within a ±8 ball window of `b` is identified:

```
R_b = { r_i : |balls_left_i − b| ≤ 8 },  r_i = y_i − ŷ_i
```

The predictive distribution is then constructed by displacing this conditional residual set onto the point prediction:

```
P = ŷ_new + R_b
```

This yields a non-parametric predictive distribution that inherits the heteroscedastic error structure of the model — wider at higher `balls_left` values, narrower as the match approaches completion — without requiring distributional assumptions. A minimum sample threshold of 50 residuals is enforced; if unmet, the full residual set is used as a fallback.

From `P`, the following quantities are derived: posterior credible intervals (P10–P90, IQR), chase success probability, implied fair odds, and conditional MAE at the current match stage.

---

## Repository Structure

```
IPL-Run-Chase-Model/
├── final_model_build.ipynb       # Full pipeline: EDA, feature engineering, model selection, evaluation
├── model_predict.ipynb           # Inference interface: five-input match state → distribution + report
├── ipl_predict_utils.py          # Modular utility functions (feature derivation, plotting, reporting)
├── requirements.txt
└── README.md
```

> `cricket_model_xgb.pkl`, `residuals.npy`, and `balls_left_test.npy` are excluded from version control. 

---

## Reproduction

```bash
git clone https://github.com/ada-aps125/IPL-Run-Chase-Model.git
cd IPL-Run-Chase-Model
pip install -r requirements.txt
```

To run inference, open `model_predict.ipynb` and set the five match-state inputs in Cell 2:

```python
BATTING_TEAM  = 'Mumbai Indians'
CURRENT_SCORE = 72
BALLS_FACED   = 54
WICKETS_LOST  = 3
TARGET        = 175
```

Execute all cells to generate the predictive distribution plot and quantitative output report.

From the above conditions and example output is included below:

```python
point_pred             = get_prediction(pipeline, state)
pred_dist, cond_resids = build_prediction_distribution(point_pred, state, residuals, balls_left_test)
metrics                = compute_metrics(pred_dist, state['target'], cond_resids)

plot_distribution(pred_dist, point_pred, state, metrics)
```

![model-distribution-plot](image.png)

```python
print_betting_report(point_pred, state, metrics)
```

════════════════════════════════════════════════════
  BETTING INSIGHTS REPORT
  Mumbai Indians  chasing 175
════════════════════════════════════════════════════

  PREDICTION
  Point estimate    : 161 runs
  Median (P50)      : 162 runs
  80% range         : 141 – 180 runs
  50% range (IQR)   : 155 – 169 runs
  Model uncertainty : ± 12.0 runs  (cond. MAE at 66 balls left)

  WIN PROBABILITY
  Chase success     : 14.7%
  Fall short        : 85.3%
  Implied odds (w/l): 6.80  /  1.17
  Within ±5 of line : 12.9%  (close finish risk)

  RUNS MARKET BREAKDOWN
  Under 155             25.4%  ███████
  155–174               59.9%  █████████████████
  175–194               11.9%  ███
  Over 194               2.8%  

  MOMENTUM
  Current RR        : 8.00
  Required RR       : 9.36
  Differential      : -1.36
  Assessment        : 🟠  Negative — bowling has the edge

  RISK STATEMENT
    No major risk flags at this stage

════════════════════════════════════════════════════


---

## Limitations & Caveats

- The model operates on aggregated team-level features and has no representation of individual player form, pitch conditions, weather, or dew factor — all of which are known to influence T20 outcomes
- Prediction intervals are widest early in the innings (balls_left > 90); estimates at this stage should be interpreted with appropriate scepticism
- The empirical distribution method assumes stationarity of residual structure across seasons; distributional shift arising from rule changes, franchise restructuring, or evolving playing styles is not modelled
- Target encoding of `batting_team` encodes historical average performance; teams with limited historical data (e.g. recently introduced franchises) may yield less reliable encoded values

---

*Python 3.8 · XGBoost · scikit-learn · category-encoders · pandas · NumPy · SciPy · matplotlib*
