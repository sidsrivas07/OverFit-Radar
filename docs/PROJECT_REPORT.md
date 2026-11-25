# Overfit Radar: development research report

This report describes the archived `data/models_v1` experiment. All results are
**under assumed cost levels**. It does not report a final-test result or establish
future profitability. The current app and accounting refactor have additional
regression checks; archived experiments have not been rerun under the new source hash.

## Research question and scope

Can IS-only features predict whether a long-or-flat daily equity strategy has
positive net OOS Sharpe, and do those predictions improve selection over IS Sharpe,
DSR and a seeded random ranking? The frozen grid has 494 settings across six families.
The cached research snapshot has twelve large-cap symbols, not all Nifty 50 members.
Its static universe creates survivorship bias.

The archived table has 122,686 eligible strategy/symbol/window rows and 24 generated
windows. Model evaluation uses eleven eligible primary development windows. The
baseline run has fourteen primary folds and eighty-four family holdout folds; the
model run has eighty-eight eligible folds across primary, family and symbol holdout
experiments. These are supervised forward folds, not CSCV partitions.

## Cost and execution assumptions

The frozen `assumed-v1` YAML contains dated approximate inputs. It is not a verified
history of Indian statutory charges. Low/base/high multiply the assumed proportional
costs by 0.5/1/2, including slippage. These multipliers are not claims about actual
institutional or retail tax schedules. Headline DP fees are excluded. A separate
DP sensitivity uses fixed sell notionals of ₹10,000, ₹50,000 and ₹1,00,000.

Signals are computed at the close, with ordinary executions at the next open.
Each evaluated IS and OOS window starts flat for its first bar. Execution begins
from the second open and any remaining position closes at the last close with
exit costs. Indicators and signal state may use earlier prices; execution positions
are reset. A last-day buy followed by closing liquidation counts as two executions.

The app and dataset now share this boundary accounting. The app's Buy & Hold uses
the same adjusted price basis, execution dates and assumed costs. Demo exploration
is restricted to dates before the reserved cutoff, 2022-01-01. Its charts are IS
illustrations, not independent OOS evaluations.

## Statistical methods

Sharpe uses per-period mean divided by sample standard deviation; annualized Sharpe
multiplies by sqrt(252). PSR uses the nonannualized Sharpe, sample length, skewness
and Pearson/non-excess kurtosis:

```
PSR = Phi((SR_hat - SR*) * sqrt(T-1)
          / sqrt(1 - skew*SR_hat + (kurtosis-1)*SR_hat**2/4))
```

DSR evaluates PSR at an expected-maximum-Sharpe benchmark computed from the variance
of trial Sharpes and number of trials. The app now measures that variance from the
configured grid on the same IS bars and cost level as the selected strategy. A
selected off-grid setting is added once. Raw N is this declared search set; it does
not account for unrecorded research choices. Constant selected returns do not receive
a PSR/DSR verdict.

Effective N uses average-linkage hierarchical clustering with distance
`sqrt(0.5 * (1 - correlation))`, cut at the configured correlation threshold
(default 0.5). Constant return series are excluded from that estimate. This is a
heuristic measure of trial dependence, not an exact independence count and not the
parameter-neighbourhood feature.

Neighbourhood features compute IS Sharpe mean, sample standard deviation and selected
Sharpe minus local mean across the candidate plus adjacent grid settings. A plateau
or spike is descriptive; neither proves robustness or overfitting.

CSCV/PBO is a separate selection-process diagnostic: equal time blocks are recombined,
the IS winner is chosen using an explicit performance metric, and its OOS rank is
converted to a logit. The implementation defaults to Sharpe, supports mean ranking,
and marks capped runs approximate. The supervised dataset itself is built using
rolling IS/embargo/OOS windows, not CSCV.

## Supervised validation

IS windows use 756 bars, OOS windows 252, a 21-bar within-row gap and 126-bar steps.
Adjacent labels overlap. The primary ML split permits training rows only if their
OOS end is strictly before test OOS start. Family holdout is a secondary experiment.
The latest available mature training window is reserved for sigmoid calibration;
fitting labels must end before that calibration window begins. Early folds without
enough fitting/calibration history are skipped and recorded.

Models are regularized logistic regression and LightGBM with fixed settings and
no evaluation-driven hyperparameter search. The full and neighbourhood-ablated
versions use matched folds. Transformations fit only on fitting rows. An explicit
feature whitelist excludes OOS targets, symbol identifiers and dates.

Metrics first average symbols (and secondary held-out families) within a window,
then average windows. Confidence intervals resample two-window circular blocks
with 2,000 seeded replicates. This accounts for adjacent-window overlap but may not
capture longer regime dependence. The final period is still unevaluated.

## Measured development results

Primary experiment, base assumed costs, full calibrated models:

| Method | AUC | Top-10 survival | Mean top-10 OOS Sharpe | Brier |
|---|---:|---:|---:|---:|
| IS Sharpe | 0.5497 | 65.68% | 0.4007 | — |
| Raw DSR | 0.5505 | 65.45% | 0.4020 | — |
| Random | 0.5012 | 63.03% | 0.3674 | — |
| Logistic | 0.5598 | 66.06% | 0.4088 | 0.2743 |
| LightGBM | 0.5464 | 66.06% | 0.3790 | 0.2780 |

LightGBM AUC interval: 0.4941–0.5961. Its paired survival difference versus IS Sharpe
is +0.00379 (95% interval −0.02273 to +0.02879). Its mean OOS Sharpe difference is
−0.02178 (interval −0.14049 to +0.08351). These results do not establish superiority.
The recorded metrics do not include OOS drawdown, so no drawdown-improvement claim
is made. No measured 68% gross-to-net failure claim is established by these outputs.

Calibration increased observed Brier from 0.2598 to 0.2780 for LightGBM and from
0.2451 to 0.2743 for logistic regression. Constant probability 0.5 has Brier 0.25
regardless of binary labels. These observed comparisons do not alone establish
statistical significance, but they do rule out claiming that calibration has
already demonstrated reliable probabilities. The next experiment script includes
constant-0.5 and fitting-set-prevalence references; these are not retrospectively
inserted into the archived results.

The SHAP snapshot explains one last-development LightGBM fit on 200 sampled rows,
in raw margins before calibration. `effective_n` has mean absolute contribution
0.4013; neighbourhood mean Sharpe has 0.1653. Spike excess has 0.00180. Absolute
values do not establish positive/negative direction, causality or consistency over
folds. DSR is not a learned survival probability.

## Reproduction and delivery status

README contains complete commands using fresh versioned directories. JSON summaries
are bundled; raw prices, training parquet, predictions and trained model binaries
are excluded. Their absence does not imply training never happened; it means the
checkout is a results/demo distribution rather than a complete binary experiment archive.

Dataset gates compare configuration, source and data hashes plus dataset bytes.
The generating Git commit is provenance; documentation-only commits no longer
invalidate unchanged inputs. A source refactor still requires a fresh versioned
dataset before a new training run. Archived manifests remain untouched.

The app provides measured IS audits, parameter surfaces and filtered research tables.
It does not currently perform trained-model inference on a chosen stock. Returns-CSV
uploads and PDF exports are not implemented. An audit can be exported as JSON.

## Limitations

Survivorship bias, approximate costs, assumed slippage, dependence among trials,
regime changes, small number of development windows, single-window calibration,
and incomplete original download timestamps. All conclusions apply to the tested
strategies and periods under assumed cost levels. Final evaluation remains a
separate, one-time step after research choices are frozen.

## References

- [Bailey & López de Prado, Deflated Sharpe Ratio](https://davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [Bailey et al., Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
