"""Research demonstrator: explicit assumed costs and IS-only statistical audits."""
from __future__ import annotations
import json
import os
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from overfit_radar.config import get_project_root,load_costs_config,load_yaml
from overfit_radar.costs.engine import BacktestEngine
from overfit_radar.costs.windows import run_window
from overfit_radar.data.validator import validate_ohlcv
from overfit_radar.strategies.base import StrategyConfig
from overfit_radar.strategies.zoo import compute_strategy_signal,build_strategy_zoo
from overfit_radar.stats.basic import calculate_sharpe,calculate_max_drawdown
from overfit_radar.report.audit import audit_strategy

st.set_page_config(page_title='Overfit Radar | Research Auditor',page_icon='🛡️',layout='wide')
ROOT=get_project_root()
RESULTS=Path(os.environ.get('OVERFIT_RESULTS_DIR',str(ROOT/'data/models_v1')))
PROTOCOL=load_yaml(ROOT/'configs/protocol.yaml')
FINAL_START=pd.Timestamp(PROTOCOL['splits']['final_start'])
st.markdown('<style>' + (Path(__file__).with_name('style.css')).read_text() + '</style>', unsafe_allow_html=True)


def hero(kicker: str, title: str, subtitle: str, status: str) -> None:
    """A compact overview for each research surface."""
    st.markdown(
        f'<div class="radar-hero"><div><div class="radar-kicker">{escape(kicker)}</div>'
        f'<div class="radar-title">{escape(title)}</div><div class="radar-sub">{escape(subtitle)}</div>'
        f'</div><div class="radar-status">{escape(status)}</div></div>',
        unsafe_allow_html=True,
    )


def dark_axes(ax, fig) -> None:
    """Use the same legible dark palette for all embedded research charts."""
    fig.patch.set_facecolor('#111b2d');ax.set_facecolor('#111b2d')
    ax.tick_params(colors='#a7bac8',labelsize=9)
    ax.xaxis.label.set_color('#a7bac8');ax.yaxis.label.set_color('#a7bac8')
    ax.title.set_color('#eaf4fa')
    for spine in ax.spines.values():spine.set_color('#30485e')
    ax.grid(color='#344d63',alpha=.35,linewidth=.7)



def read_json(path: Path):
    """Read on rerun so newly created artifacts are not hidden by stale caches."""
    return json.loads(path.read_text()) if path.is_file() else None


def research_results():
    hero('Research / archived evaluation','Development results','Measured across the sample universe; independent of the selected stock.','FINAL PERIOD RESERVED')
    st.caption('Saved experiments across the sample universe; independent of the stock selected in the sidebar. All results are under assumed cost levels. The final period is unevaluated.')
    rows=read_json(RESULTS/'summary.json')
    if rows is None:
        st.info('No saved model summary found. Run the documented experiment pipeline to generate it.')
        return
    summary=pd.DataFrame(rows)
    experiment=st.selectbox('Research experiment',sorted(summary.experiment.unique()),index=sorted(summary.experiment.unique()).index('primary'))
    level=st.selectbox('Research cost level',['base','low','high'])
    metric=st.selectbox('Research metric',['auc','survival_rate','mean_oos_sharpe','brier','ece','spearman'])
    selected=summary[(summary.experiment==experiment)&(summary.cost_level==level)&(summary.metric==metric)]
    st.dataframe(selected[['method','windows','mean','ci_low','ci_high']].reset_index(drop=True))
    st.caption('Intervals resample blocks of two adjacent windows. Top-k uses ten strategies per symbol/window. AUC describes ranking; it does not establish probability calibration.')
    base=summary[(summary.experiment=='primary')&(summary.cost_level=='base')]
    probability=base[base.metric.isin(['brier','ece']) & base.method.isin(['lightgbm_full_raw','lightgbm_full_calibrated','logistic_full_raw','logistic_full_calibrated'])]
    st.subheader('Probability quality: raw versus calibrated')
    st.dataframe(probability[['method','metric','mean','ci_low','ci_high']].reset_index(drop=True))
    st.info('For this saved run, calibration increased observed Brier scores for both full models. Constant probability 0.5 has Brier 0.25 for any binary outcomes. A fitted calibrator does not guarantee reliable probabilities.')
    predictions_path=RESULTS/'predictions.parquet'
    if predictions_path.is_file():
        predictions=pd.read_parquet(predictions_path)
        predictions=predictions[predictions.experiment==experiment]
        label='survived_'+level
        if label in predictions:
            st.subheader('Reliability diagram')
            fig,ax=plt.subplots(figsize=(7,4));dark_axes(ax,fig);ax.plot([0,1],[0,1],'--',color='#8aa0b2',label='Perfect agreement')
            for method in ['lightgbm_full_raw','lightgbm_full_calibrated','logistic_full_raw','logistic_full_calibrated']:
                if method not in predictions:continue
                points=predictions[[method,label]].dropna().copy()
                points['bin']=np.minimum((points[method]*10).astype(int),9)
                grouped=points.groupby('bin')[[method,label]].mean()
                ax.plot(grouped[method],grouped[label],marker='o',label=method)
            ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed survival fraction');ax.legend(fontsize=7,facecolor='#17253a',edgecolor='#30485e',labelcolor='#eaf4fa')
            st.pyplot(fig);plt.close(fig)
            st.caption('Descriptive pooled-row reliability diagram with ten equal-width bins. Correlated rows are not independent; reported metric intervals use window blocks.')
    else:
        st.caption('Row-level prediction arrays are not bundled, so a reliability curve cannot be reconstructed from these summaries. A fresh model run saves the arrays needed to display it.')
    paired=read_json(RESULTS/'paired_comparisons.json')
    if paired:
        paired=pd.DataFrame(paired)
        filtered=paired[(paired.experiment==experiment)&(paired.cost_level==level)&(paired.metric==metric)]
        if not filtered.empty:
            st.subheader('Paired differences on matching windows')
            st.dataframe(filtered[['model','reference','difference','ci_low','ci_high']].reset_index(drop=True))
    importance=read_json(RESULTS/'shap_importance.json')
    if importance:
        st.subheader('SHAP: last development LightGBM fit')
        st.caption('Mean absolute contributions on 200 sampled rows, in raw log-odds before calibration. Magnitudes do not show direction or consistency across folds. effective_n measures dependence among trials, not parameter-neighbourhood stability.')
        importance=pd.DataFrame(importance).head(10).iloc[::-1]
        fig,ax=plt.subplots(figsize=(10,4));dark_axes(ax,fig);ax.barh(importance.feature,importance.mean_absolute_shap,color='#49dbc4')
        ax.set_xlabel('Mean absolute SHAP contribution (raw log-odds)');st.pyplot(fig);plt.close(fig)


@st.cache_data(show_spinner=False)
def measured_audit(prices,selected,zoo,costs,cost_multiplier,start,end):
    return audit_strategy(prices,selected,zoo,BacktestEngine(costs,cost_multiplier=cost_multiplier),start,end)

st.sidebar.markdown('<div class="radar-kicker">◈ QUANT RESEARCH LAB</div>',unsafe_allow_html=True)
st.sidebar.title('Overfit Radar')
st.sidebar.caption('Strategy robustness · Indian equities')
tabs=st.tabs(['◉  Overview','▦  Parameter map','◇  Statistical audit','▥  Research'])
with tabs[3]:research_results()
paths=sorted((ROOT/'data/raw').glob('*.parquet'))
if not paths:
    for tab in tabs[:3]:
        with tab:
            st.info('No cached prices found. Saved research benchmarks remain available in the Research benchmarks tab.')
            st.code('python scripts/download_sample_universe.py',language='bash')
    st.stop()

symbol=st.sidebar.selectbox('Target sample stock',[p.stem for p in paths])
try:
    prices=pd.read_parquet(ROOT/'data/raw'/f'{symbol}.parquet')
    prices.index=pd.to_datetime(prices.index)
    if prices.index.tz is not None:prices.index=prices.index.tz_localize(None)
    validate_ohlcv(prices,symbol)
except (ValueError,KeyError,OSError) as error:
    st.error(f'Cannot use this price cache: {error}');st.stop()
# The demo never exposes reserved final-period prices for strategy selection.
prices=prices[prices.index<FINAL_START]
if len(prices)<2:
    st.warning('At least two development-period bars are required. Reserved final-period prices are excluded.');st.stop()
period=st.sidebar.date_input('In-sample audit period',value=(prices.index[max(0,len(prices)-756)].date(),prices.index[-1].date()),min_value=prices.index[0].date(),max_value=prices.index[-1].date())
if len(period)!=2:
    st.info('Choose both a start and an end date.');st.stop()
start=int(prices.index.searchsorted(pd.Timestamp(period[0]),side='left'))
end=int(prices.index.searchsorted(pd.Timestamp(period[1]),side='right'))-1
if end-start<1:
    st.warning('Choose a period containing at least two trading bars.');st.stop()
family=st.sidebar.selectbox('Strategy family',['SMA Crossover','Time Series Momentum','RSI Mean Reversion'])
if family=='SMA Crossover':
    fast=st.sidebar.slider('Fast window (days)',5,50,15)
    slow=st.sidebar.slider('Slow window (days)',20,200,50)
    if fast>=slow:
        with tabs[0]:st.warning('Fast window must be smaller than slow window. Adjust the parameters to continue.')
        st.stop()
    selected=StrategyConfig('selected','sma_crossover',dict(fast_window=fast,slow_window=slow))
elif family=='Time Series Momentum':
    lookback=st.sidebar.slider('Lookback (days)',10,252,63);holding=st.sidebar.slider('Holding period (days)',1,42,5)
    selected=StrategyConfig('selected','time_series_momentum',dict(lookback=lookback,holding_period=holding))
else:
    window=st.sidebar.slider('RSI window (days)',5,21,14)
    oversold=st.sidebar.slider('Oversold level',20,40,30);overbought=st.sidebar.slider('Overbought level',60,80,70)
    selected=StrategyConfig('selected','rsi_mean_reversion',dict(window=window,oversold=oversold,overbought=overbought))
level=st.sidebar.selectbox('Assumed cost level',['base','low','high'])
multiplier=float(PROTOCOL['cost_levels'][level]);costs=load_costs_config();engine=BacktestEngine(costs,cost_multiplier=multiplier)
st.sidebar.caption(f'{level.upper()} · {multiplier:g}× assumed-v1. Headline DP fees excluded.')
history=prices.iloc[:end+1]
signal=compute_strategy_signal(history,selected)
result=run_window(engine,history,signal,start,end)
# Buy-and-hold uses the same adjusted prices, entry date, liquidation and assumed fees.
benchmark=run_window(engine,history,pd.Series(1.,index=history.index),start,end)
net_sharpe=calculate_sharpe(result.net_return)
with tabs[0]:
    hero('Strategy / backtest',f'{family} · {symbol.replace("_NS",".NS")}',f'{result.index[0].date()} — {result.index[-1].date()} · {level.upper()} assumed costs','IS EXPLORATION')
    st.markdown('<div class="radar-note">Exploratory IS chart. First bar flat; later trades use the next open; final holdings close with exit costs. Earlier prices may warm indicators. This is not an OOS prediction.</div>',unsafe_allow_html=True)
    st.markdown('<div class="radar-section">Performance snapshot</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric('Gross Sharpe',f'{calculate_sharpe(result.gross_return):.2f}')
    b.metric('Net Sharpe',f'{net_sharpe:.2f}')
    c.metric('Max drawdown',f'{calculate_max_drawdown(result.net_return):.1%}')
    d.metric('Completed trades',str(int(result.sell_count.sum())))
    st.caption(f'Annual turnover: {(result.buy_count.sum()+result.sell_count.sum())/len(result)*252:.1f}× · under assumed cost levels')
    st.markdown('<div class="radar-section">Equity curves</div>',unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(12,4));dark_axes(ax,fig)
    for returns,label in [(result.gross_return,'Gross strategy'),(result.net_return,'Net strategy'),(benchmark.net_return,'Buy & hold · adjusted / same costs')]:
        color={'Gross strategy':'#74a8d8','Net strategy':'#49dbc4','Buy & hold · adjusted / same costs':'#d9b56e'}[label]
        ax.plot(returns.index,(1+returns).cumprod(),label=label,color=color,linewidth=1.8 if label!='Net strategy' else 2.3)
    ax.set_ylabel('Wealth from 1 unit');ax.legend(facecolor='#17253a',edgecolor='#30485e',labelcolor='#eaf4fa',fontsize=8,loc='upper left');fig.tight_layout();st.pyplot(fig,width="stretch");plt.close(fig)

with tabs[1]:
    hero('Diagnostics / parameter surface','Plateau or spike?',f'{family} · {symbol.replace("_NS",".NS")}', 'IS ONLY')
    st.caption('Neighbourhood performance is a diagnostic. A plateau does not establish an edge, and a spike alone does not prove overfitting.')
    if family!='SMA Crossover':st.info('The parameter surface currently supports SMA crossover.')
    else:
        fast_range=[x for x in [fast-4,fast-2,fast,fast+2,fast+4] if x>=5]
        slow_range=[x for x in [slow-10,slow-5,slow,slow+5,slow+10] if x>=2]
        surface=np.full((len(fast_range),len(slow_range)),np.nan)
        for i,f in enumerate(fast_range):
            for j,s in enumerate(slow_range):
                if f>=s:continue
                cfg=StrategyConfig('neighbour','sma_crossover',dict(fast_window=f,slow_window=s))
                res=run_window(engine,history,compute_strategy_signal(history,cfg),start,end)
                surface[i,j]=calculate_sharpe(res.net_return)
        if not np.isfinite(surface).any():st.info('No valid neighbourhood configurations are available.')
        else:
            fig,ax=plt.subplots(figsize=(8,4));dark_axes(ax,fig);img=ax.matshow(np.ma.masked_invalid(surface),cmap='RdYlGn',vmin=-2,vmax=2);cb=fig.colorbar(img);cb.ax.tick_params(colors='#a7bac8')
            ax.set_xticks(range(len(slow_range)));ax.set_xticklabels(slow_range);ax.set_yticks(range(len(fast_range)));ax.set_yticklabels(fast_range)
            ax.set_xlabel('Slow window');ax.set_ylabel('Fast window');fig.tight_layout();st.pyplot(fig,width="stretch");plt.close(fig)
            a,b=st.columns(2);a.metric('Neighbourhood mean Sharpe',f'{np.nanmean(surface):.2f}');b.metric('Sharpe above neighbourhood mean',f'{net_sharpe-np.nanmean(surface):.2f}')

with tabs[2]:
    hero('Statistics / multiple testing','Measured PSR & DSR','Trial variance is computed from the configured grid for this IS period.', 'IS ONLY')
    st.write('The audit measures per-period Sharpe variance across the configured strategy grid on this same IS period and assumed cost level. An off-grid selected configuration is added once. The count does not include unrecorded research trials.')
    if st.button('Compute measured audit'):
        with st.spinner('Evaluating the declared trial grid…'):
            audit=measured_audit(history,selected,build_strategy_zoo(),costs,multiplier,start,end)
        st.caption(f"{audit['raw_n']} trials; {audit['observations']} bars; {audit['period_start']} to {audit['period_end']}; daily Sharpe variance {audit['variance']:.8g}.")
        st.caption(f"Cluster-based effective N: {audit['effective_n']:.2f}. The effective count is an IS-only heuristic.")
        if not audit['estimable']:st.warning('The selected strategy has no measurable return variation in this period. PSR/DSR are not reported.')
        else:
            a,b,c=st.columns(3);a.metric('PSR',f"{audit['psr']:.1%}");b.metric('DSR, raw N',f"{audit['dsr_raw']:.1%}");c.metric('DSR, effective N',f"{audit['dsr_effective']:.1%}")
            if audit['dsr_raw']>=.95:st.info('Meets the 95% raw-N DSR threshold under the declared assumptions. This is not proof of future profitability.')
            else:st.info('Does not meet the 95% raw-N DSR threshold under the declared assumptions. This does not prove the strategy is a false discovery.')
        st.caption('DSR is a statistical multiple-testing diagnostic, not the probability of positive OOS Sharpe.')
        st.download_button('Download audit details',json.dumps(audit,indent=2),file_name='is_audit.json',mime='application/json')
    else:st.info('Run the audit to compute the trial variance and counts. No assumed variance or verdict is shown before calculation.')
