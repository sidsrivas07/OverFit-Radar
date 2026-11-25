"""Shared flat-start and terminal-liquidation accounting for app and dataset."""
from __future__ import annotations
from typing import Tuple
import pandas as pd


def window_returns(prices: pd.DataFrame, net: pd.DataFrame, costs: pd.DataFrame,
                   actions: pd.DataFrame, positions: pd.DataFrame,
                   start: int, end: int, engine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Slice inclusive bars, reset execution, and close at the final close.

    Signals may already use prior-price warm-up. First bar is flat. The terminal
    Series records extra close sells independently of opening trade actions, so
    a final-day buy plus liquidation correctly counts as two executions.
    """
    if start < 0 or end >= len(prices) or end-start < 1:
        raise ValueError('A window needs at least two bars within price history')
    net=net.iloc[start:end+1].copy();costs=costs.iloc[start:end+1].copy()
    actions=actions.iloc[start:end+1].copy();positions=positions.iloc[start:end+1].copy()
    net.iloc[0]=0.;costs.iloc[0]=0.;actions.iloc[0]=0.
    factor=prices.adjusted_close/prices.close
    intraday=prices.adjusted_close.iloc[start+1]/(prices.open.iloc[start+1]*factor.iloc[start+1])-1
    buy=engine.cost_model.compute_cost_rate(prices.index[start+1],True,engine.slippage_bps,engine.cost_multiplier)
    target=positions.iloc[1]
    actions.iloc[1]=target;costs.iloc[1]=target*buy;net.iloc[1]=target*(intraday-buy)
    terminal=positions.iloc[-1].copy()
    sell=engine.cost_model.compute_cost_rate(prices.index[end],False,engine.slippage_bps,engine.cost_multiplier)
    if engine.dp_trade_notional is not None:
        sell+=engine.cost_model.get_rates_for_date(prices.index[end]).dp_charges_per_sell/engine.dp_trade_notional*engine.cost_multiplier
    costs.iloc[-1]+=terminal*sell;net.iloc[-1]-=terminal*sell
    return net,costs,actions,terminal


def run_window(engine, prices: pd.DataFrame, signal: pd.Series, start: int = 0,
               end: int | None = None) -> pd.DataFrame:
    """Execute a single strategy using the dataset's window accounting policy."""
    end=len(prices)-1 if end is None else end
    if not signal.index.equals(prices.index):
        raise ValueError('Signal and price indices must match exactly')
    continuous=engine.run(prices.iloc[:end+1],signal.iloc[:end+1])
    frames=[continuous[c].to_frame('strategy') for c in ['net_return','cost','trade_action','position']]
    net,cost,actions,terminal=window_returns(prices,*frames,start,end,engine)
    result=continuous.iloc[start:end+1].copy()
    result['net_return']=net.iloc[:,0];result['cost']=cost.iloc[:,0]
    result['gross_return']=result.net_return+result.cost
    result['trade_action']=actions.iloc[:,0]
    result.iloc[0,result.columns.get_loc('position')]=0.
    result['forced_exit']=0.
    result.iloc[-1,result.columns.get_loc('forced_exit')]=terminal.iloc[0]
    result.iloc[-1,result.columns.get_loc('position')]=0.
    result['buy_count']=(result.trade_action==1).astype(int)
    result['sell_count']=(result.trade_action==-1).astype(int)+result.forced_exit.astype(int)
    return result
