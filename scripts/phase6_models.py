"""Development-only ML experiments; dataset gates and final cutoff are mandatory.

Training code/config are fingerprinted independently of the unchanged dataset
builder. Run with DYLD_LIBRARY_PATH pointing at sklearn/.dylibs on this Mac.
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from overfit_radar.features.provenance import load_for_model_training, digest
from overfit_radar.config import get_project_root, load_yaml
from overfit_radar.eval.splits import forward_folds
from overfit_radar.eval.metrics import ranking_metrics
from overfit_radar.models.baselines import baseline_scores

NUMERIC = ['is_sharpe','is_sortino','is_max_drawdown','is_skewness','is_kurtosis',
           'is_autocorr','is_psr','is_dsr_raw','is_dsr_eff','effective_n',
           'family_trials_count','is_trade_count','is_annual_turnover',
           'is_sharpe_2x_cost','is_cost_sensitivity','is_observations']
NEIGHBOUR = ['neighbourhood_sharpe_mean','neighbourhood_sharpe_std','neighbourhood_spike_excess']


def calibration_partition(train: pd.DataFrame, config: dict):
    """Reserve latest mature window; purge fitting labels crossing its OOS start."""
    latest = train.oos_start_date.max()
    cal = train[train.oos_start_date == latest]
    fit = train[train.oos_end_date < latest]
    if (fit.window_id.nunique() < config['min_fit_windows'] or len(fit) < config['min_fit_rows']
        or len(cal) < config['min_calibration_rows'] or fit.survived.nunique()<2 or cal.survived.nunique()<2):
        return None
    if not fit.oos_end_date.max() < cal.oos_start_date.min():
        raise ValueError('Calibration leakage')
    return fit, cal


def feature_frame(table: pd.DataFrame, neighbourhood: bool) -> pd.DataFrame:
    """Whitelist IS columns; never infer predictors from all remaining columns."""
    columns = NUMERIC + (NEIGHBOUR if neighbourhood else []) + ['family']
    frame=table[columns].copy()
    frame[columns[:-1]]=frame[columns[:-1]].replace([np.inf,-np.inf],np.nan)
    return frame


def logit(probability):
    p=np.clip(probability,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def fit_model(fit, cal, kind, neighbourhood, config):
    """Fit preprocessing/classifier on fit rows, sigmoid calibrator on cal rows."""
    columns=NUMERIC+(NEIGHBOUR if neighbourhood else [])
    transform=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),columns),
        ('family',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['family'])])
    if kind=='logistic':
        classifier=LogisticRegression(random_state=config['seed'],**config['logistic'])
    else:
        from lightgbm import LGBMClassifier
        classifier=LGBMClassifier(random_state=config['seed'],**config['lightgbm'])
    pipeline=Pipeline([('transform',transform),('classifier',classifier)])
    pipeline.fit(feature_frame(fit,neighbourhood),fit.survived)
    raw=pipeline.predict_proba(feature_frame(cal,neighbourhood))[:,1]
    calibrator=LogisticRegression(C=1e6,max_iter=1000,solver='liblinear',random_state=config['seed'])
    calibrator.fit(logit(raw),cal.survived)
    return pipeline,calibrator


def probability_metrics(y,p,bins=10):
    """AUC plus Brier, log loss and equal-width expected calibration error."""
    y=np.asarray(y);p=np.asarray(p)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
    ids=np.minimum((p*bins).astype(int),bins-1)
    ece=sum(np.mean(ids==i)*abs(y[ids==i].mean()-p[ids==i].mean()) for i in range(bins) if np.any(ids==i))
    return dict(auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),ece=float(ece))


def probability_references(fit: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Constant and fitting-label prevalence references on the evaluation row index."""
    return pd.DataFrame({'constant_05': 0.5, 'fit_prevalence': float(fit.survived.mean())},index=index)


def bootstrap_summary(windows,config):
    keys=['experiment','cost_level','method'];metrics=['survival_rate','mean_oos_sharpe','spearman','auc','brier','ece']
    summaries=[];paired=[]
    for key,g in windows.groupby(keys,sort=True):
        g=g.sort_values('window_id');n=len(g);block=config['bootstrap']['block_length']
        rng=np.random.default_rng(config['seed'])
        indices=(rng.integers(n,size=(config['bootstrap']['repetitions'],int(np.ceil(n/block)),1))+np.arange(block))%n
        indices=indices.reshape(len(indices),-1)[:,:n]
        for metric in metrics:
            a=g[metric].to_numpy()
            if np.isfinite(a).sum()==0:continue
            samples=np.nanmean(a[indices],axis=1)
            summaries.append(dict(zip(keys,key),metric=metric,windows=n,mean=float(np.nanmean(a)),ci_low=float(np.nanquantile(samples,.025)),ci_high=float(np.nanquantile(samples,.975))))
    # Paired window differences use the same resampled windows for both methods.
    for (experiment,level),g in windows.groupby(['experiment','cost_level']):
        for metric in ['survival_rate','mean_oos_sharpe','spearman','brier','ece']:
            wide=g.pivot(index='window_id',columns='method',values=metric).sort_index()
            models=[c for c in wide if c.endswith('_calibrated')]
            for model in models:
                references=['is_sharpe','is_dsr_raw','random'] if metric not in ['brier','ece'] else ['constant_05','fit_prevalence']
                if '_full_' in model:references += [model.replace('_full_','_ablated_')]
                for reference in references:
                    if reference not in wide:continue
                    a=(wide[model]-wide[reference]).dropna().to_numpy();n=len(a)
                    if n<2:continue
                    rng=np.random.default_rng(config['seed']);block=config['bootstrap']['block_length']
                    indices=(rng.integers(n,size=(config['bootstrap']['repetitions'],int(np.ceil(n/block)),1))+np.arange(block))%n
                    samples=a[indices.reshape(len(indices),-1)[:,:n]].mean(axis=1)
                    paired.append(dict(experiment=experiment,cost_level=level,model=model,reference=reference,metric=metric,windows=n,difference=float(a.mean()),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975))))
    return summaries,paired


def run(dataset: Path, output: Path, config_path: Path):
    """Run fixed development experiments without tuning on evaluation labels."""
    if output.exists():raise FileExistsError('Choose a fresh experiment output directory')
    table=load_for_model_training(dataset)
    config=yaml.safe_load(config_path.read_text());protocol=load_yaml(get_project_root()/'configs/protocol.yaml')
    if config['final_start']!=protocol['splits']['final_start']:raise ValueError('Final cutoff mismatch')
    table=table[table.oos_end_date < pd.Timestamp(config['final_start'])].copy()
    output.mkdir(parents=True)
    manifest=dict(status='running',dataset_sha256=digest(dataset),dataset_manifest_sha256=digest(dataset.with_suffix('.manifest.json')),
                  experiment_code_sha256=digest(Path(__file__)),experiment_config_sha256=digest(config_path),config=config,
                  python=platform.python_version(),packages={p:metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','shap']},
                  claim='under assumed cost levels',final_period_evaluated=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));(output/'model_experiment.yaml').write_text(config_path.read_text())
    calendar=pd.DatetimeIndex(sorted(set().union(*(set(pd.read_parquet(p).index) for p in (get_project_root()/'data/raw').glob('*.parquet')))))
    predictions=[];metrics=[];folds=[];skips=[];last={}
    for experiment in ['primary','family_holdout','universe_holdout']:
        for fold in forward_folds(table,calendar,config['final_start'],leave_one_family_out=experiment=='family_holdout'):
            train=table.iloc[fold.train].copy();test=table.iloc[fold.test].copy()
            if experiment=='universe_holdout':
                train=train[~train.symbol.isin(config['heldout_symbols'])];test=test[test.symbol.isin(config['heldout_symbols'])]
            partition=calibration_partition(train,config)
            fold_id=f'{experiment}_{fold.family}_{fold.prediction_date.date()}'
            if partition is None or test.empty:
                skips.append(dict(fold=fold_id,reason='Insufficient purged fitting/calibration history'));continue
            fit,cal=partition
            assert fit.oos_end_date.max()<cal.oos_start_date.min()
            assert cal.oos_end_date.max()<test.oos_start_date.min()
            assert test.oos_end_date.max()<pd.Timestamp(config['final_start'])
            folds.append(dict(fold=fold_id,experiment=experiment,family=fold.family,fit_rows=len(fit),fit_windows=fit.window_id.nunique(),calibration_rows=len(cal),test_rows=len(test),fit_label_end=str(fit.oos_end_date.max()),calibration_start=str(cal.oos_start_date.min()),calibration_label_end=str(cal.oos_end_date.max()),test_start=str(test.oos_start_date.min()),test_end=str(test.oos_end_date.max())))
            print(f'{fold_id}: fit={len(fit)}, cal={len(cal)}, test={len(test)}',flush=True)
            scores=baseline_scores(test)
            references=probability_references(fit,test.index)
            for name in references:scores[name]=references[name]
            probability_methods=['constant_05','fit_prevalence']
            for kind in ['logistic','lightgbm']:
                for neighbour in [True,False]:
                    name=f'{kind}_{"full" if neighbour else "ablated"}'
                    pipeline,calibrator=fit_model(fit,cal,kind,neighbour,config)
                    raw=pipeline.predict_proba(feature_frame(test,neighbour))[:,1]
                    scores[name+'_raw']=raw;scores[name+'_calibrated']=calibrator.predict_proba(logit(raw))[:,1]
                    probability_methods.extend([name+'_raw',name+'_calibrated'])
                    if experiment=='primary':last[name]=(pipeline,calibrator,test.copy(),neighbour)
            ids=['symbol','window_id','strategy_id','family','oos_start_date','oos_end_date']
            pred=test[ids+['survived_'+level for level in protocol['cost_levels']]].copy()
            pred['experiment']=experiment;pred['heldout_family']=fold.family
            for method in scores:pred[method]=scores[method]
            predictions.append(pred)
            for level in protocol['cost_levels']:
                labelled=test.assign(oos_sharpe=test[f'oos_sharpe_{level}'],survived=test[f'survived_{level}'])
                for method in scores:
                    m=ranking_metrics(labelled,scores[method],protocol['top_k'])
                    m['experiment']=experiment;m['cost_level']=level;m['method']=method;m['heldout_family']=fold.family
                    for column in ['auc','brier','log_loss','ece']:m[column]=np.nan
                    if method in probability_methods:
                        for i,row in m.iterrows():
                            mask=(labelled.symbol==row.symbol)&(labelled.window_id==row.window_id)
                            for key,value in probability_metrics(labelled.loc[mask,'survived'],scores.loc[mask,method],config['calibration_bins']).items():m.loc[i,key]=value
                    if method not in probability_methods:
                        for i,row in m.iterrows():
                            mask=(labelled.symbol==row.symbol)&(labelled.window_id==row.window_id)
                            if labelled.loc[mask,'survived'].nunique()==2:
                                m.loc[i,'auc']=roc_auc_score(labelled.loc[mask,'survived'],scores.loc[mask,method])
                    metrics.append(m)
    if not predictions:raise RuntimeError('No eligible model folds')
    predictions=pd.concat(predictions,ignore_index=True);predictions.to_parquet(output/'predictions.parquet',index=False)
    raw=pd.concat(metrics,ignore_index=True);raw.to_parquet(output/'per_symbol_metrics.parquet',index=False)
    measures=['survival_rate','mean_oos_sharpe','spearman','auc','brier','log_loss','ece']
    windows=raw.groupby(['experiment','cost_level','method','window_id'])[measures].mean().reset_index()
    windows.to_json(output/'per_window_metrics.json',orient='records',indent=2)
    summary,paired=bootstrap_summary(windows,config)
    (output/'summary.json').write_text(json.dumps(summary,indent=2));(output/'paired_comparisons.json').write_text(json.dumps(paired,indent=2))
    (output/'folds.json').write_text(json.dumps(folds,indent=2));(output/'skipped_folds.json').write_text(json.dumps(skips,indent=2))
    # Explain the predetermined last development full LightGBM fit in raw log-odds.
    import shap,joblib
    pipeline,calibrator,test,neighbour=last['lightgbm_full']
    sample=test.sample(min(config['shap_sample_rows'],len(test)),random_state=config['seed'])
    transformed=pipeline.named_steps['transform'].transform(feature_frame(sample,True))
    names=pipeline.named_steps['transform'].get_feature_names_out()
    explainer=shap.TreeExplainer(pipeline.named_steps['classifier'])
    values=explainer.shap_values(transformed)
    if isinstance(values,list):values=values[1]
    if np.asarray(values).ndim==3:values=values[:,:,1]
    base=explainer.expected_value
    if np.asarray(base).size>1:base=np.asarray(base)[1]
    margin=pipeline.named_steps['classifier'].booster_.predict(transformed,raw_score=True)
    np.testing.assert_allclose(np.asarray(values).sum(axis=1)+float(np.asarray(base).reshape(-1)[0]),margin,rtol=1e-5,atol=1e-5)
    sample[['symbol','window_id','strategy_id']].to_json(output/'shap_sample.json',orient='records',indent=2)
    importance=pd.DataFrame(dict(feature=names,mean_absolute_shap=np.abs(values).mean(axis=0))).sort_values('mean_absolute_shap',ascending=False)
    importance.to_json(output/'shap_importance.json',orient='records',indent=2)
    np.savez_compressed(output/'shap_values.npz',values=values,features=transformed,names=names,expected_value=base,raw_margin=margin)
    for name,(pipeline,calibrator,_,neighbour) in last.items():
        joblib.dump(dict(pipeline=pipeline,calibrator=calibrator,neighbourhood=neighbour),output/(name+'.joblib'))
    manifest.update(status='complete',eligible_folds=len(folds),skipped_folds=len(skips),primary_windows=int(windows[windows.experiment=='primary'].window_id.nunique()),
                    shap_scope='Last development primary full LightGBM; raw margin, not calibrated probability; 200 sampled rows',
                    artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));print('COMPLETE',manifest['primary_windows'],'primary windows',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path,default=Path(__file__).with_name('model_experiment.yaml'))
    args=parser.parse_args();run(args.dataset,args.output,args.config)
