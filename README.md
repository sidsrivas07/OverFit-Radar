# Overfit Radar

A research tool for auditing long-or-flat daily equity backtests. It compares
in-sample statistics and machine learning rankings with later strategy outcomes.
All performance claims are **under assumed cost levels**. This is not a live
trading service or a reconstruction of historical Indian transaction costs.

## What is implemented

- A configured grid of 494 strategy settings across six families.
- Daily execution at the next open; each evaluated window starts flat and closes
  remaining positions at its last close with exit costs. Indicators may use prior prices.
- PSR, DSR with raw and cluster-based effective trial counts, and a separate CSCV/PBO diagnostic.
- IS-only neighbourhood, risk and execution features; later OOS survival labels.
- Forward model splits, separate chronological calibration, family and symbol holdouts,
  neighbourhood ablation, cost sensitivity, and bootstrap over windows.
- Streamlit backtests, parameter surfaces, a measured IS DSR audit and saved research summaries.

The app does not currently load a model to predict survival for the selected stock.
Its model tab displays archived research results. Uploaded-returns CSV analysis and
PDF report export are future functionality; an IS audit can be downloaded as JSON.

## Setup and demo

Use a virtual environment. Python 3.9+ is supported by the project metadata;
archived model results record Python 3.12.13. Local fix validation used Python 3.9.6.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m streamlit run app/app.py
```

Saved JSON research summaries are included and can be viewed without market data.
Raw prices, the training parquet, model binaries and prediction arrays are not
included. The app shows setup instructions instead of crashing when prices are absent.
To enable the stock backtest and audit tabs:

```bash
.venv/bin/python scripts/download_sample_universe.py
```

This downloads a **twelve-symbol sample**, not the complete Nifty 50 universe.
The research uses a static large-cap universe and is subject to survivorship bias.
The app restricts exploration to dates before the reserved final cutoff, 2022-01-01.

The statistical audit evaluates the configured grid on the chosen IS period and
cost level. An off-grid selection is added once. It measures daily Sharpe variance,
raw N and effective N; it does not assume a fixed variance. DSR is a multiple-testing
diagnostic, not a next-period survival probability. The declared N does not capture
unrecorded experiments. The effective count is a heuristic.

## Reproduce a fresh experiment

Choose new directories for each version. Do not overwrite the archived results.
The baseline command regenerates the dataset and creates its manifest and gates;
`generate_training_dataset.py` is a legacy unversioned helper and is not needed here.

```bash
.venv/bin/python scripts/download_sample_universe.py
.venv/bin/python scripts/run_phase6_baselines.py --output data/runs/baselines-v2
.venv/bin/python scripts/run_models.py --dataset data/runs/baselines-v2/training.parquet --output data/runs/models-v2
```

`run_models.py` forwards to the training script and, on macOS, uses scikit-learn's
bundled OpenMP directory when available. If the installed wheels provide no usable
OpenMP runtime, follow the official LightGBM installation instructions for the platform.
The new experiment includes constant-0.5 and fitting-set-prevalence probability
references. Their new results require running the experiment; they have not been
inserted into archived metrics.

To display a fresh result directory:

```bash
OVERFIT_RESULTS_DIR=data/runs/models-v2 .venv/bin/python -m streamlit run app/app.py
```

Dataset manifests record source/data/config hashes, cost version, generating commit
and local snapshot date. Original download timestamps are unavailable for the archived
snapshot. Input or dataset changes are rejected before training. Git commit identity
is recorded as provenance; a documentation-only commit does not invalidate unchanged
research inputs. Source refactoring in this fix changes the fingerprint, so a future
training run must regenerate its versioned table. Archived results remain historical
outputs from their recorded implementation, not newly rerun results.

The primary split requires training OOS end < test OOS start. Family holdout is
secondary. Calibration fitting is also purged chronologically. The final period
remains unevaluated. Parameters and protocol must be frozen before any eventual final run.

## Saved development results

The archived dataset contains 122,686 eligible rows from twelve cached symbols and
24 generated windows. The baseline evaluation has 14 primary folds plus 84 family
holdout folds. Models have 11 eligible primary windows; 88 model folds include
primary, family-holdout and symbol-holdout experiments. These are rolling forward
folds, **not CSCV partitions**.

Primary experiment, base assumed costs, full calibrated models:

| Method | AUC | Top-10 survival | Mean top-10 OOS Sharpe | Brier |
|---|---:|---:|---:|---:|
| IS Sharpe ranking | 0.5497 | 65.68% | 0.4007 | — |
| Raw DSR ranking | 0.5505 | 65.45% | 0.4020 | — |
| Seeded random ranking | 0.5012 | 63.03% | 0.3674 | — |
| Logistic regression | 0.5598 | 66.06% | 0.4088 | 0.2743 |
| LightGBM | 0.5464 | 66.06% | 0.3790 | 0.2780 |

Metrics first average symbols within each window, then windows. LightGBM AUC's
95% block-bootstrap interval is 0.4941–0.5961. Its survival difference versus IS
Sharpe is +0.00379, with paired interval −0.02273 to +0.02879. The saved evidence
does not establish superiority over the baseline.

Calibration increased observed Brier scores: LightGBM 0.2598 → 0.2780; logistic
0.2451 → 0.2743. Constant probability 0.5 has Brier 0.25 for every binary outcome.
Fitting a calibrator does not establish trustworthy probabilities. Negative and
inconclusive results are part of the research outcome.

The SHAP snapshot explains the last primary LightGBM fit on 200 sampled rows,
in raw log-odds before calibration. `effective_n` is the largest mean absolute
contribution (0.4013); it measures dependence among trial returns, not parameter
neighbourhood stability. Absolute attribution does not establish direction or causality.

## Tests and environment

```bash
# Offline unit, regression and app tests
.venv/bin/python -m pytest -m 'not integration' -p no:cacheprovider
# After downloading the sample universe
.venv/bin/python -m pytest -m integration -p no:cacheprovider
```

The market-cache integration test is explicitly skipped if no cache exists. A
partially populated or invalid cache still fails validation. Tests cover an empty
app cache, invalid SMA settings, measured DSR inputs, future-price invariance, window
reset/liquidation (including a final-day entry), and provenance rejection.

`requirements-tested-py39.txt` records exact versions used by the local macOS/Python
3.9 regression environment. It is not a universal lockfile or the archived training
environment. The training manifest records versions used for the saved research run.

## Limitations

Static-universe survivorship bias; assumed costs and slippage; correlated trials;
regime dependence; a single calibration window; incomplete provider provenance;
and limited development-window count. Two-window bootstrap blocks account for
adjacent overlap but may not capture longer dependence. Results concern the tested
strategies and periods under assumed cost levels, not future profitability.

## References

- [Bailey & López de Prado: Deflated Sharpe Ratio](https://davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [Bailey et al.: Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- [LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)
