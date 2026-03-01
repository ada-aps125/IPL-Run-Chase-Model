"""
ipl_predict_utils.py
────────────────────
Utility functions for the IPL Chase Predictor notebook.
Import at the top of model_predict.ipynb — no logic needed in the notebook cells.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde


# ── Style constants ────────────────────────────────────────────────────────────
STYLE = dict(
    BG='#0F1117',
    PANEL='#181C25',
    ACCENT='#4FC3F7',
    HIGHLIGHT='#FFD54F',
    DANGER='#FF6B6B',
    SUCCESS='#69F0AE',
    GRID='#262B38',
    TEXT='#E0E6F0',
    SUBTEXT='#7A8499',
)

TEAMS = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bengaluru',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Rajasthan Royals',
    'Sunrisers Hyderabad', 'Punjab Kings', 'Lucknow Super Giants',
    'Gujarat Titans'
]


def apply_plot_style() -> None:
    """Apply dark notebook-wide matplotlib style."""
    plt.rcParams.update({
        'figure.facecolor': STYLE['BG'],   'axes.facecolor': STYLE['PANEL'],
        'axes.edgecolor':   STYLE['GRID'], 'axes.labelcolor': STYLE['TEXT'],
        'axes.spines.top':  False,         'axes.spines.right': False,
        'axes.spines.left': False,         'axes.spines.bottom': False,
        'xtick.color':      STYLE['SUBTEXT'], 'ytick.color': STYLE['SUBTEXT'],
        'xtick.labelsize':  10,            'ytick.labelsize': 10,
        'grid.color':       STYLE['GRID'], 'grid.linewidth': 0.6,
        'font.family':      'DejaVu Sans',
    })


# ── Feature engineering ────────────────────────────────────────────────────────

def build_match_state(batting_team: str, current_score: int,
                      balls_faced: int, wickets_lost: int,
                      target: int) -> dict:
    """
    Derive all model features and summary stats from raw match inputs.

    Parameters
    ----------
    batting_team  : str  — chasing team name
    current_score : int  — runs scored so far
    balls_faced   : int  — balls faced so far (1–119)
    wickets_lost  : int  — wickets fallen (0–9)
    target        : int  — runs target to chase

    Returns
    -------
    dict with keys: balls_left, overs_faced, current_rr, runs_needed,
                    required_rr, wickets_in_hand, + all raw inputs
    """
    _validate_inputs(batting_team, current_score,
                     balls_faced, wickets_lost, target)

    overs_faced = balls_faced / 6
    balls_left = 120 - balls_faced
    current_rr = current_score / overs_faced
    runs_needed = target - current_score
    required_rr = runs_needed / \
        (balls_left / 6) if balls_left > 0 else float('inf')
    wickets_in_hand = 10 - wickets_lost

    return dict(
        batting_team=batting_team,
        current_score=current_score,
        balls_faced=balls_faced,
        wickets_lost=wickets_lost,
        target=target,
        overs_faced=overs_faced,
        balls_left=balls_left,
        current_rr=current_rr,
        runs_needed=runs_needed,
        required_rr=required_rr,
        wickets_in_hand=wickets_in_hand,
    )


def build_model_input(state: dict) -> pd.DataFrame:
    """Convert a match state dict into a single-row DataFrame for the pipeline."""
    return pd.DataFrame({
        'balls_left': [state['balls_left']],
        'wickets': [state['wickets_in_hand']],
        'batting_team': [state['batting_team']],
        'current_rr': [state['current_rr']],
        'required_rr': [state['required_rr']],
    })


def print_match_summary(state: dict) -> None:
    """Print a formatted match situation summary."""
    s = state
    print('Match Situation Summary')
    print('─' * 40)
    print(f"  Batting Team  : {s['batting_team']}")
    print(f"  Score         : {s['current_score']}/{s['wickets_lost']}  "
          f"off {s['balls_faced']} balls  ({s['overs_faced']:.1f} overs)")
    print(f"  Target        : {s['target']}")
    print(f"  Runs Needed   : {s['runs_needed']}  "
          f"off {s['balls_left']} balls  ({s['balls_left']/6:.1f} overs)")
    print(f"  Current RR    : {s['current_rr']:.2f}")
    print(f"  Required RR   : {s['required_rr']:.2f}")
    print(f"  Wickets Left  : {s['wickets_in_hand']}")


# ── Prediction & distribution ──────────────────────────────────────────────────

def get_prediction(pipeline, state: dict) -> float:
    """Return the point prediction from the loaded pipeline."""
    return float(pipeline.predict(build_model_input(state))[0])


def build_prediction_distribution(point_pred: float, state: dict,
                                  residuals: np.ndarray,
                                  balls_left_test: np.ndarray,
                                  tolerance: int = 8) -> np.ndarray:
    """
    Construct an empirical predictive distribution by shifting conditional
    residuals onto the point prediction.

    Falls back to the full residual array if fewer than 50 conditional
    samples exist within `tolerance` balls of the current game stage.
    """
    mask = np.abs(balls_left_test - state['balls_left']) <= tolerance
    cond_residuals = residuals[mask]

    if len(cond_residuals) < 50:
        print(f'⚠  Only {len(cond_residuals)} conditional samples — '
              f'falling back to full residual distribution.')
        cond_residuals = residuals

    return point_pred + cond_residuals, cond_residuals


def compute_metrics(pred_dist: np.ndarray, target: int,
                    cond_residuals: np.ndarray) -> dict:
    """
    Compute all probabilistic and betting metrics from the distribution.

    Returns
    -------
    dict with keys: p10, p25, p50, p75, p90, prob_chase, prob_lose,
                    prob_within_5, implied_odds_win, implied_odds_lose,
                    cond_mae, cond_std, bands
    """
    p10, p25, p50, p75, p90 = np.percentile(pred_dist, [10, 25, 50, 75, 90])
    prob_chase = float(np.mean(pred_dist >= target))
    prob_lose = 1 - prob_chase
    prob_within_5 = float(np.mean(np.abs(pred_dist - target) <= 5))

    bands = [
        (f'Under {target - 20}',
         float(np.mean(pred_dist < target - 20))),
        (f'{target-20}–{target-1}',
         float(np.mean((pred_dist >= target - 20) & (pred_dist < target)))),
        (f'{target}–{target+19}',
         float(np.mean((pred_dist >= target) & (pred_dist < target + 20)))),
        (f'Over {target + 19}',
         float(np.mean(pred_dist >= target + 20))),
    ]

    return dict(
        p10=p10, p25=p25, p50=p50, p75=p75, p90=p90,
        prob_chase=prob_chase,
        prob_lose=prob_lose,
        prob_within_5=prob_within_5,
        implied_odds_win=1 / prob_chase if prob_chase > 0 else float('inf'),
        implied_odds_lose=1 / prob_lose if prob_lose > 0 else float('inf'),
        cond_mae=float(np.mean(np.abs(cond_residuals))),
        cond_std=float(np.std(cond_residuals)),
        bands=bands,
    )


def momentum_label(current_rr: float, required_rr: float) -> str:
    """Return an emoji-tagged momentum assessment string."""
    diff = current_rr - required_rr
    if diff > 2.0:
        return '🟢  Strong — batting well above required rate'
    elif diff > 0.5:
        return '🟡  Positive — slight edge to batting'
    elif diff > -0.5:
        return '⚪  Neutral — evenly poised'
    elif diff > -2.0:
        return '🟠  Negative — bowling has the edge'
    else:
        return '🔴  Critical — batting under severe pressure'


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_distribution(pred_dist: np.ndarray, point_pred: float,
                      state: dict, metrics: dict, save_path=None) -> None:
    """
    Plot the KDE predictive distribution with annotated zones,
    credible interval, target line, and prediction line.
    """
    s, m = state, metrics
    C = STYLE

    kde = gaussian_kde(pred_dist, bw_method=0.3)
    x_range = np.linspace(pred_dist.min() - 10, pred_dist.max() + 10, 500)
    kde_vals = kde(x_range)

    fig = plt.figure(figsize=(13, 6), facecolor=C['BG'])
    gs = GridSpec(1, 1, figure=fig, left=0.08,
                  right=0.96, top=0.82, bottom=0.13)
    ax = fig.add_subplot(gs[0])

    # Fill zones
    ax.fill_between(x_range, kde_vals, where=(x_range >= s['target']),
                    color=C['SUCCESS'], alpha=0.20,
                    label=f"Chase zone  (≥{s['target']})")
    ax.fill_between(x_range, kde_vals, where=(x_range < s['target']),
                    color=C['DANGER'],  alpha=0.15,
                    label=f"Fall short  (<{s['target']})")
    ax.fill_between(x_range, kde_vals,
                    where=((x_range >= m['p25']) & (x_range <= m['p75'])),
                    color=C['ACCENT'], alpha=0.25, label='50% credible interval')

    ax.plot(x_range, kde_vals, color=C['ACCENT'], linewidth=2.2, zorder=4)
    ax.axvline(point_pred,   color=C['HIGHLIGHT'], linewidth=2.0,
               label=f"Model prediction: {point_pred:.0f}")
    ax.axvline(s['target'],  color='white',        linewidth=1.4, linestyle='--',
               label=f"Target: {s['target']}")

    for pval, plabel in [(m['p10'], 'P10'), (m['p90'], 'P90')]:
        ax.axvline(pval, color=C['SUBTEXT'],
                   linewidth=0.9, linestyle=':', zorder=3)
        ax.text(pval, ax.get_ylim()[1] * 0.02, plabel,
                color=C['SUBTEXT'], fontsize=8, ha='center', va='bottom')

    ax.yaxis.grid(True, zorder=0)
    ax.set_xlabel('Predicted Final Score (runs)',
                  labelpad=10, fontsize=11, color=C['TEXT'])
    ax.set_ylabel('Probability Density',          labelpad=10,
                  fontsize=11, color=C['TEXT'])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

    fig.text(0.08, 0.92,
             f"{s['batting_team']}  |  {s['current_score']}/{s['wickets_lost']} "
             f"off {s['balls_faced']} balls  —  Chasing {s['target']}",
             fontsize=13, fontweight='bold', color=C['TEXT'], ha='left')
    fig.text(0.08, 0.87,
             f"Predicted: {point_pred:.0f}  |  "
             f"80% range: {m['p10']:.0f}–{m['p90']:.0f}  |  "
             f"Chase probability: {m['prob_chase']*100:.1f}%",
             fontsize=10, color=C['SUBTEXT'], ha='left')

    ax.legend(frameon=True, framealpha=0.15, edgecolor=C['GRID'],
              labelcolor=C['TEXT'], fontsize=9, loc='upper left')
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=STYLE['BG'])
    plt.show()


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_betting_report(point_pred: float, state: dict, metrics: dict) -> None:
    """Print the full structured betting insights report."""
    s, m = state, metrics
    rr_diff = s['current_rr'] - s['required_rr']
    div = '═' * 52

    print(f'\n{div}')
    print(f"  BETTING INSIGHTS REPORT")
    print(f"  {s['batting_team']}  chasing {s['target']}")
    print(div)

    print(f'\n  PREDICTION')
    print(f"  Point estimate    : {point_pred:.0f} runs")
    print(f"  Median (P50)      : {m['p50']:.0f} runs")
    print(f"  80% range         : {m['p10']:.0f} – {m['p90']:.0f} runs")
    print(f"  50% range (IQR)   : {m['p25']:.0f} – {m['p75']:.0f} runs")
    print(f"  Model uncertainty : ± {m['cond_mae']:.1f} runs  "
          f"(cond. MAE at {s['balls_left']} balls left)")

    print(f'\n  WIN PROBABILITY')
    print(f"  Chase success     : {m['prob_chase']*100:.1f}%")
    print(f"  Fall short        : {m['prob_lose']*100:.1f}%")
    print(
        f"  Implied odds (w/l): {m['implied_odds_win']:.2f}  /  {m['implied_odds_lose']:.2f}")
    print(
        f"  Within ±5 of line : {m['prob_within_5']*100:.1f}%  (close finish risk)")

    print(f'\n  RUNS MARKET BREAKDOWN')
    for band_label, prob in m['bands']:
        bar = '█' * int(prob * 30)
        print(f'  {band_label:<20} {prob*100:5.1f}%  {bar}')

    print(f'\n  MOMENTUM')
    print(f"  Current RR        : {s['current_rr']:.2f}")
    print(f"  Required RR       : {s['required_rr']:.2f}")
    print(f"  Differential      : {rr_diff:+.2f}")
    print(
        f"  Assessment        : {momentum_label(s['current_rr'], s['required_rr'])}")

    print(f'\n  RISK STATEMENT')
    flags = []
    if m['prob_within_5'] > 0.20:
        flags.append(f"  High close-finish probability ({m['prob_within_5']*100:.0f}%) "
                     f"— line markets are volatile")
    if m['cond_std'] > 20:
        flags.append(f"  High model uncertainty at this stage "
                     f"(std={m['cond_std']:.1f}) — wide outcome spread")
    if s['wickets_lost'] >= 6:
        flags.append('  Tail-heavy lower distribution likely '
                     '— lower-order batting unpredictable')
    if s['balls_left'] > 90:
        flags.append(f"  Very early in the chase "
                     f"— model error is ~{m['cond_mae']:.0f} runs at this stage")
    for f in flags:
        print(f)
    if not flags:
        print('    No major risk flags at this stage')

    print(f'\n{div}\n')


# ── Validation (private) ───────────────────────────────────────────────────────

def _validate_inputs(batting_team, current_score, balls_faced,
                     wickets_lost, target) -> None:
    assert batting_team in TEAMS, \
        f"Team not recognised. Options: {TEAMS}"
    assert 0 < balls_faced < 120, \
        "BALLS_FACED must be between 1 and 119"
    assert 0 <= wickets_lost <= 9, \
        "WICKETS_LOST must be between 0 and 9"
    assert target > current_score, \
        "TARGET must be greater than CURRENT_SCORE"
