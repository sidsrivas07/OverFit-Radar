"""Regression checks for IS audit evidence, execution and empty-data app flows."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
from overfit_radar.config import get_project_root,load_costs_config
from overfit_radar.costs.engine import BacktestEngine
from overfit_radar.costs.windows import run_window,window_returns
from overfit_radar.strategies.zoo import build_strategy_zoo,compute_strategy_signal
from overfit_radar.strategies.base import StrategyConfig
from overfit_radar.report.audit import audit_strategy
from overfit_radar.stats.basic import calculate_sharpe


def test_window_execution_matches_independent_reset_oracle(synthetic_gbm_prices):
    prices=synthetic_gbm_prices(n_days=400)
    for multiplier in [.5,1.,2.]:
        engine=BacktestEngine(load_costs_config(),cost_multiplier=multiplier,dp_trade_notional=10000)
        for start,end in [(0,100),(133,399)]:
            for strategy in build_strategy_zoo()[::70]:
                signal=compute_strategy_signal(prices,strategy)
                expected=engine.run(prices.iloc[start:end+1],signal.iloc[start:end+1])
                forced=expected.position.iloc[-1]
                cost=engine.cost_model.compute_cost_rate(prices.index[end],False,engine.slippage_bps,multiplier)
                cost+=engine.cost_model.get_rates_for_date(prices.index[end]).dp_charges_per_sell/10000*multiplier
                expected.loc[expected.index[-1],'cost']+=forced*cost
                expected.loc[expected.index[-1],'net_return']-=forced*cost
                actual=run_window(engine,prices,signal,start,end)
                np.testing.assert_allclose(actual.net_return,expected.net_return,atol=1e-14)
                np.testing.assert_allclose(actual.cost,expected.cost,atol=1e-14)
                assert actual.position.iloc[0]==actual.position.iloc[-1]==0
                assert actual.buy_count.sum()==actual.sell_count.sum()


def test_last_day_entry_counts_both_buy_and_close(synthetic_gbm_prices):
    prices=synthetic_gbm_prices(n_days=10);signal=pd.Series(0.,index=prices.index);signal.iloc[-2]=1.
    engine=BacktestEngine(load_costs_config())
    result=run_window(engine,prices,signal)
    assert result.buy_count.sum()==result.sell_count.sum()==1
    assert result.forced_exit.iloc[-1]==1
    day=prices.index[-1]
    expected=sum(engine.cost_model.compute_cost_rate(day,buy,engine.slippage_bps) for buy in [True,False])
    assert result.cost.iloc[-1]==pytest.approx(expected)


def test_audit_uses_measured_trial_variance_and_no_future(synthetic_gbm_prices):
    prices=synthetic_gbm_prices(n_days=400);zoo=build_strategy_zoo()[:4]
    engine=BacktestEngine(load_costs_config())
    result=audit_strategy(prices,zoo[1],zoo,engine,start=30,end=299)
    sharpe=[calculate_sharpe(run_window(engine,prices,compute_strategy_signal(prices,s),30,299).net_return,annualize=False) for s in zoo]
    assert result['raw_n']==4 and result['selected_in_grid']
    assert result['variance']==pytest.approx(np.var(sharpe,ddof=1))
    changed=prices.copy();changed.iloc[300:,:5]*=5
    assert audit_strategy(changed,zoo[1],zoo,engine,30,299)==result
    custom=StrategyConfig('custom','sma_crossover',dict(fast_window=9,slow_window=37))
    assert audit_strategy(prices,custom,zoo,engine,30,299)['raw_n']==5


@pytest.fixture
def app_context(tmp_path,monkeypatch):
    from streamlit import cache_data
    root=get_project_root();shutil.copytree(root/'configs',tmp_path/'configs')
    shutil.copytree(root/'data/models_v1',tmp_path/'data/models_v1')
    (tmp_path/'data/raw').mkdir(parents=True)
    monkeypatch.setattr('overfit_radar.config.get_project_root',lambda:tmp_path)
    cache_data.clear()
    yield root/'app/app.py',tmp_path
    cache_data.clear()


def test_app_empty_cache_keeps_benchmarks_available(app_context):
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(app_context[0])).run(timeout=30)
    assert not app.exception
    assert any('No cached prices' in x.value for x in app.info)
    assert len(app.dataframe)>0


def test_app_rejects_invalid_sma_and_renders_valid_audit(app_context,synthetic_gbm_prices,monkeypatch):
    from streamlit.testing.v1 import AppTest
    root,path=app_context
    prices=synthetic_gbm_prices(n_days=400,start_date='2010-01-04')
    prices.to_parquet(path/'data/raw/TEST_NS.parquet')
    monkeypatch.setattr('overfit_radar.strategies.zoo.build_strategy_zoo',lambda: [StrategyConfig('sma','sma_crossover',dict(fast_window=15,slow_window=50))])
    app=AppTest.from_file(str(root)).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any(x.label=='DSR, raw N' for x in app.metric)
    for x in app.slider:
        if x.label=='Fast window (days)':x.set_value(50)
        if x.label=='Slow window (days)':x.set_value(20)
    app.run(timeout=30)
    assert not app.exception
    assert any('Fast window must be smaller' in x.value for x in app.warning)
    assert not any(x.value=='nan' for x in app.metric)


def test_documentation_commit_does_not_invalidate_data(tmp_path,monkeypatch):
    from overfit_radar.features import provenance as p
    fingerprint={'commit':'old','commit_status':'available','config_hash':'stable','source_hashes':{'builder':'same'}}
    monkeypatch.setattr(p,'fingerprint',lambda:fingerprint.copy())
    path=p.save_versioned(pd.DataFrame({'x':[1]}),tmp_path)
    fingerprint['commit']='docs-only'
    assert len(p.load_training_table(path))==1
    fingerprint['source_hashes']={'builder':'changed'}
    with pytest.raises(ValueError,match='source_hashes'):p.load_training_table(path)
