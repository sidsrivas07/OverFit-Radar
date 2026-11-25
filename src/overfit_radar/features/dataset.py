"""Dataset builder orchestrating rolling walk-forward window partitioning, feature extraction, and labeling."""

from __future__ import annotations

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from overfit_radar.config import CostsConfig, ExperimentConfig
from overfit_radar.costs.engine import BacktestEngine
from overfit_radar.costs.windows import window_returns
from overfit_radar.features.extractor import extract_window_features
from overfit_radar.features.neighbourhood import build_neighbourhood_graph
from overfit_radar.features.windows import generate_walk_forward_windows, WindowSlice
from overfit_radar.stats.basic import calculate_sharpe
from overfit_radar.strategies.base import StrategyConfig
from overfit_radar.strategies.zoo import build_strategy_zoo, compute_strategy_signal

logger = logging.getLogger(__name__)


def build_strategy_dataset(
    universe_data: Dict[str, pd.DataFrame],
    zoo: Optional[List[StrategyConfig]] = None,
    costs_config: Optional[CostsConfig] = None,
    experiment_config: Optional[ExperimentConfig] = None,
) -> pd.DataFrame:
    """
    Construct the master supervised training dataset across symbols and rolling walk-forward windows.
    
    Row schema:
    - strategy_id: Unique strategy identifier
    - family: Strategy family
    - symbol: Stock ticker (e.g. 'RELIANCE.NS')
    - window_id: Walk-forward partition identifier
    - is_start_date, is_end_date, oos_start_date, oos_end_date
    - In-Sample features: is_sharpe, is_sortino, is_max_drawdown, is_skewness, is_kurtosis,
                         is_autocorr, is_psr, is_dsr_raw, is_dsr_eff, effective_n,
                         neighbourhood_sharpe_mean, neighbourhood_sharpe_std, neighbourhood_spike_excess
    - Out-of-Sample targets:
      - oos_sharpe: Realized net annualized Sharpe ratio in OOS
      - survived: Binary label (1 if oos_sharpe > 0.0, else 0)
    """
    from overfit_radar.config import load_costs_config, load_experiment_config
    
    if zoo is None:
        zoo = build_strategy_zoo()
    if costs_config is None:
        costs_config = load_costs_config()
    if experiment_config is None:
        experiment_config = load_experiment_config()

    if costs_config.cost_multiplier != 1.0:
        raise ValueError("Build the training table at 1x costs; use the explicit 2x features for stress tests")
    from overfit_radar.config import get_project_root, load_yaml
    protocol = load_yaml(get_project_root() / "configs/protocol.yaml")
    wf_cfg = experiment_config.walk_forward
    engine = BacktestEngine(costs_config, cost_multiplier=costs_config.cost_multiplier)
    neighbourhood_graph = build_neighbourhood_graph(zoo)

    all_rows: List[Dict] = []

    for sym, price_df in universe_data.items():
        logger.info(f"Generating backtest returns for {sym} ({len(price_df)} bars)...")
        # Run entire zoo on this symbol once: returns matrix shape (T, N)
        from overfit_radar.data.validator import validate_ohlcv
        validate_ohlcv(price_df, sym)
        ids = [s.strategy_id for s in zoo]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Zoo must have nonempty, unique strategy IDs")
        shape = (len(price_df), len(zoo))
        values = {name: np.empty(shape, dtype=np.float64) for name in ("net", "cost", "action", "position")}
        for j, strat in enumerate(zoo):
            result = engine.run(price_df, compute_strategy_signal(price_df, strat))
            for name, column in (("net", "net_return"), ("cost", "cost"), ("action", "trade_action"), ("position", "position")):
                values[name][:, j] = result[column].to_numpy()
        full_returns = pd.DataFrame(values["net"], index=price_df.index, columns=ids)
        full_costs = pd.DataFrame(values["cost"], index=price_df.index, columns=ids)
        full_positions = pd.DataFrame(values["position"], index=price_df.index, columns=ids)
        full_actions = pd.DataFrame(values["action"], index=price_df.index, columns=ids)

        # Generate rolling walk-forward windows
        windows = generate_walk_forward_windows(
            dates=price_df.index,
            is_months=wf_cfg.in_sample_months,
            oos_months=wf_cfg.out_of_sample_months,
            step_months=wf_cfg.step_months,
            embargo_days=wf_cfg.embargo_days,
        )
        logger.info(f"Processing {len(windows)} rolling walk-forward windows for {sym}...")

        for w in windows:
            rets_is, costs_is, actions_is, terminal_is = window_returns(
                price_df, full_returns, full_costs, full_actions, full_positions,
                w.is_start_idx, w.is_end_idx, engine)
            rets_oos, costs_oos, _, _ = window_returns(
                price_df, full_returns, full_costs, full_actions, full_positions,
                w.oos_start_idx, w.oos_end_idx, engine)

            # Extract In-Sample Features
            features_df = extract_window_features(
                returns_is=rets_is,
                zoo=zoo,
                costs_config=costs_config,
                neighbourhood_graph=neighbourhood_graph,
            )


            # Compute Out-of-Sample Labels
            for strat in zoo:
                sid = strat.strategy_id
                if sid not in features_df.index:
                    continue

                actions = actions_is[sid]
                # Count round trips opened AND closed in this IS slice. An inherited
                # position's first exit does not constitute a completed IS trade.
                entries = int((actions > 0).sum())
                exits = int((actions < 0).sum())
                nonzero = actions[actions != 0]
                inherited_exit = int(not nonzero.empty and nonzero.iloc[0] < 0)
                completed = min(entries, exits - inherited_exit + int(terminal_is[sid]))
                if completed < wf_cfg.min_is_trades:
                    continue
                oos_s_rets = rets_oos[sid]
                oos_sharpe = calculate_sharpe(oos_s_rets, annualize=True)
                survived = 1 if oos_sharpe > experiment_config.survival_threshold else 0

                feat_dict = features_df.loc[sid].to_dict()
                sr_2x = calculate_sharpe(rets_is[sid] - costs_is[sid], annualize=True)
                feat_dict.update({
                    "params": json.dumps(strat.params, sort_keys=True),
                    "is_trade_count": completed,
                    "is_annual_turnover": float((actions.abs().sum() + terminal_is[sid]) / len(actions) * 252),
                    "is_sharpe_2x_cost": sr_2x,
                    "is_cost_sensitivity": feat_dict["is_sharpe"] - sr_2x,
                    "is_observations": len(rets_is),
                })
                row = {
                    "strategy_id": sid,
                    "symbol": sym,
                    "window_id": w.window_id,
                    "is_start_date": w.is_start_date,
                    "is_end_date": w.is_end_date,
                    "oos_start_date": w.oos_start_date,
                    "oos_end_date": w.oos_end_date,
                    **feat_dict,
                    "oos_sharpe": oos_sharpe,
                    "survived": survived,
                }
                for level, multiplier in protocol["cost_levels"].items():
                    stressed = rets_oos[sid] - (multiplier-1)*costs_oos[sid]
                    row[f"oos_sharpe_{level}"] = calculate_sharpe(stressed)
                    row[f"survived_{level}"] = int(row[f"oos_sharpe_{level}"] > experiment_config.survival_threshold)
                all_rows.append(row)

    dataset_df = pd.DataFrame(all_rows)
    logger.info(f"Successfully generated training dataset: {dataset_df.shape} rows.")
    return dataset_df
