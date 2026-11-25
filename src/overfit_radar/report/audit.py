"""IS-only strategy audit measured against an explicit candidate grid."""
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from overfit_radar.costs.windows import run_window
from overfit_radar.strategies.base import StrategyConfig
from overfit_radar.strategies.zoo import compute_strategy_signal
from overfit_radar.stats.basic import calculate_sharpe,calculate_skewness,calculate_kurtosis
from overfit_radar.stats.psr_dsr import compute_psr,compute_dsr,compute_effective_n


def audit_strategy(prices: pd.DataFrame, selected: StrategyConfig, zoo: List[StrategyConfig],
                   engine, start: int = 0, end: int | None = None) -> Dict[str, Any]:
    """Estimate variance and N from the supplied IS trial set, adding an off-grid choice.

    Only history through end is used. Counts describe this declared grid, not every
    experiment the researcher may have tried. DSR is not OOS survival probability.
    """
    end=len(prices)-1 if end is None else end
    history=prices.iloc[:end+1]
    candidates=list(zoo)
    matching=[s for s in candidates if s.family==selected.family and s.params==selected.params]
    candidate_id=matching[0].strategy_id if matching else '__selected_custom__'
    if not matching:
        candidates.append(StrategyConfig(candidate_id,selected.family,selected.params))
    if len({s.strategy_id for s in candidates})!=len(candidates):raise ValueError('Duplicate candidate IDs')
    returns={s.strategy_id:run_window(engine,history,compute_strategy_signal(history,s),start,end).net_return for s in candidates}
    matrix=pd.DataFrame(returns)
    sharpes=matrix.apply(lambda x:calculate_sharpe(x,annualize=False))
    variance=float(sharpes.var(ddof=1)) if len(sharpes)>1 else 0.
    rets=matrix[candidate_id];sr=calculate_sharpe(rets,annualize=False)
    skew=calculate_skewness(rets);kurt=calculate_kurtosis(rets)
    effective=compute_effective_n(matrix,method='cluster')
    active=bool(rets.std(ddof=1)>1e-12)
    return dict(raw_n=len(candidates),effective_n=effective,variance=variance,
                selected_in_grid=bool(matching),candidate_id=candidate_id,observations=len(rets),
                period_start=str(rets.index[0].date()),period_end=str(rets.index[-1].date()),
                estimable=active,
                psr=compute_psr(sr,0.,len(rets),skew,kurt) if active else None,
                dsr_raw=compute_dsr(sr,variance,len(candidates),len(rets),skew,kurt) if active else None,
                dsr_effective=compute_dsr(sr,variance,effective,len(rets),skew,kurt) if active else None)
