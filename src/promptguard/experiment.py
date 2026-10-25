"""Experiment orchestration: every selection operation excludes the final test."""
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .utils import write_json, fingerprint, digest, size_bytes, hardware
from .splitting import assert_isolated, group_holdout
from .tfidf import train_tfidf
from .oof import FEATURE_ORDER, generate_oof, code_fingerprint
from .ensemble import StackingEnsemble, WeightedEnsemble
from .evaluation import evaluate, benchmark, comparison
from .robustness import category_availability, disagreement, paired_cluster_bootstrap, family_holdout


def read_prepared(config):
    d=pd.read_json(Path(config['prepared_dir'])/'samples.jsonl',lines=True)
    parts=[d[d.split==name].reset_index(drop=True) for name in ['train','validation','test']]
    assert_isolated(parts)
    expected=json.loads((Path(config['prepared_dir'])/'preparation.json').read_text())['fingerprint']
    if fingerprint(d)!=expected:raise ValueError('Prepared data changed; rerun prepare and use a new run directory')
    return parts


def predict_base(model,name,texts):
    return model.predict_proba(list(texts))[:,1] if name=='tfidf' else model.predict(list(texts))


def fit_base(name,train,config,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    identity=digest(json.dumps({'data':fingerprint(train),'config':config,'model':name,'code':code_fingerprint()},sort_keys=True))
    cache=path/'complete.json'
    if cache.exists():
        info=json.loads(cache.read_text())
        if info['identity']!=identity:raise ValueError(f'Stale model cache at {path}; choose a new run directory')
        if name=='tfidf':model=joblib.load(path/'pipeline.joblib')
        else:
            from .transformer import TransformerPredictor
            model=TransformerPredictor(path/'refit/best',config['transformer']['batch_size'])
        return model,info['costs']
    if name=='tfidf':model,costs=train_tfidf(train,config,path)
    else:
        from .transformer import train_transformer,loss_plot
        model,costs=train_transformer(train,name,config,path);loss_plot(path)
    write_json(cache,{'identity':identity,'fit_ids':train.sample_id.tolist(),'fit_groups':sorted(train.group.unique()),'costs':costs})
    return model,costs


def independent(config,name):
    train,val,test=read_prepared(config);out=Path(config['run_dir']);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'config.json',config);write_json(out/'hardware.json',hardware())
    model,costs=fit_base(name,train,config,out/'models'/name)
    predict=lambda texts:predict_base(model,name,texts)
    costs={**costs,**benchmark(predict,val.text,batch_size=config['transformer']['batch_size'] if name!='tfidf' else 16)}
    modelpath=out/'models'/name/('pipeline.joblib' if name=='tfidf' else 'refit/best')
    costs['model_bytes']=size_bytes(modelpath)
    if name!='tfidf':costs['validation_truncation']=model.truncation_stats(val.text)
    result=evaluate(name,val,test,predict(val.text),predict(test.text),out/'evaluation',costs)
    category_availability(pd.concat([train,val,test]),out/'evaluation')
    write_json(out/f'{name}_inference.json',{'model':name,'path':str(modelpath.relative_to(out)),
        'threshold':result['validation']['threshold'],'max_length':config['transformer']['max_length'],
        'data_fingerprint':fingerprint(pd.concat([train,val,test]))})
    return result


def _fold_predict(name,fit,held,path,config):
    model,_=fit_base(name,fit,config,path/'model')
    scores=predict_base(model,name,held.text)
    del model;gc.collect()
    return scores


def run_oof(config,names=FEATURE_ORDER):
    train,_,_=read_prepared(config)
    return generate_oof(train,config,Path(config['run_dir'])/'oof',
        lambda name,fit,held,path:_fold_predict(name,fit,held,path,config),model_names=names)


def preflight(config):
    from .transformer import load_base
    status={}
    for name in ['distilbert','bert','roberta']:
        try:
            tokenizer,model=load_base(config['transformer']['models'][name],config['transformer'])
            status[name]={'available':True};del tokenizer,model;gc.collect()
        except (OSError,ImportError) as error:status[name]={'available':False,'error':str(error)}
    write_json(Path(config['run_dir'])/'preflight.json',{'hardware':hardware(),'models':status})
    return status


def full_experiment(config):
    out=Path(config['run_dir']);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'config.json',config)
    status=preflight(config)
    if not all(s['available'] for s in status.values()):
        raise RuntimeError('Pretrained models unavailable. See preflight.json; no Transformer or ensemble metrics generated.')
    train,val,test=read_prepared(config)
    if config['stacking_mode']=='oof':
        features=run_oof(config);meta_y=train.label.to_numpy();base_train=train;provenance='oof'
    else:
        base_train,meta=group_holdout(train,config['oof_folds'],config['seed']+27)
        assert_isolated([base_train,meta,val,test]);scores={}
        for name in FEATURE_ORDER:
            model,_=fit_base(name,base_train,config,out/'models'/name)
            scores[name]=predict_base(model,name,meta.text);del model;gc.collect()
        features=pd.DataFrame(scores);meta_y=meta.label.to_numpy();provenance='disjoint_holdout'
        write_json(out/'holdout_manifest.json',{'base_fit_ids':base_train.sample_id.tolist(),'meta_fit_ids':meta.sample_id.tolist(),
            'limitation':'Base models remain trained on base_fit only. They are NOT refit on meta rows after fitting the meta model.'})
    val_scores={};test_scores={};results=[];cost_by_model={};paths={};thresholds={}
    for name in FEATURE_ORDER:
        model,costs=fit_base(name,base_train,config,out/'models'/name)
        fn=lambda texts:predict_base(model,name,texts)
        val_scores[name]=fn(val.text);test_scores[name]=fn(test.text)
        costs={**costs,**benchmark(fn,val.text,batch_size=config['transformer']['batch_size'] if name!='tfidf' else 16)}
        p=out/'models'/name/('pipeline.joblib' if name=='tfidf' else 'refit/best')
        costs['model_bytes']=size_bytes(p);cost_by_model[name]=costs;paths[name]=str(p.relative_to(out))
        if name!='tfidf':costs['validation_truncation']=model.truncation_stats(val.text)
        result=evaluate(name,val,test,val_scores[name],test_scores[name],out/'evaluation',costs)
        thresholds[name]=result['validation']['threshold'];results.append(result)
        write_json(out/f'{name}_inference.json',{'model':name,'path':str(p.relative_to(out)),
            'threshold':thresholds[name],'max_length':config['transformer']['max_length']})
        del model;gc.collect()
    val_x=pd.DataFrame(val_scores,columns=FEATURE_ORDER);test_x=pd.DataFrame(test_scores,columns=FEATURE_ORDER)
    features=features.loc[:,list(FEATURE_ORDER)]
    definitions=[('roberta_tfidf',WeightedEnsemble(('roberta','tfidf'))),
        ('transformers',WeightedEnsemble(('bert','roberta','distilbert'))),
        ('weighted',WeightedEnsemble()),('stacking',StackingEnsemble(seed=config['seed']))]
    ensemble_paths={};ensemble_thresholds={};test_predictions={name:(np.asarray(test_scores[name])>=thresholds[name]).astype(int) for name in FEATURE_ORDER}
    for name,ensemble in definitions:
        columns=list(ensemble.order);start=time.perf_counter()
        ensemble.fit(features[columns],meta_y,provenance=provenance);meta_seconds=time.perf_counter()-start
        (out/'ensembles').mkdir(exist_ok=True);path=out/'ensembles'/f'{name}.joblib';joblib.dump(ensemble,path)
        costs={'meta_fit_seconds':meta_seconds,'base_refit_seconds':sum(cost_by_model[k]['wall_seconds'] for k in columns),
            'model_bytes':sum(cost_by_model[k]['model_bytes'] for k in columns)+size_bytes(path),
            'estimated_sequential_single_ms':sum(cost_by_model[k]['single_ms_p50'] for k in columns),
            'latency_note':'Base latency sum is an estimate; see bundle_benchmark.json for measured full inference latency.'}
        if isinstance(ensemble,WeightedEnsemble):costs['weights']=dict(zip(columns,ensemble.weights.tolist()))
        v=ensemble.predict(val_x[columns]);t=ensemble.predict(test_x[columns])
        result=evaluate(name,val,test,v,t,out/'evaluation',costs);results.append(result)
        ensemble_thresholds[name]=result['validation']['threshold'];ensemble_paths[name]=str(path.relative_to(out))
        test_predictions[name]=(t>=ensemble_thresholds[name]).astype(int)
    comparison(results,out/'evaluation');disagreement(test,test_x,thresholds,out/'evaluation')
    write_json(out/'evaluation/paired_bootstrap.json',paired_cluster_bootstrap(test,test_predictions))
    category_availability(pd.concat([train,val,test]),out/'evaluation')
    # Record OOF selection + fit cost, separate from final fit and meta cost.
    fold_costs=[]
    for p in (out/'oof').glob('fold_*/*/model/complete.json'):
        fold_costs.append(json.loads(p.read_text())['costs']['wall_seconds'])
    write_json(out/'evaluation/compute_summary.json',{'oof_base_fit_seconds':sum(fold_costs),
        'oof_fit_count':len(fold_costs),'final_base_fit_seconds':sum(c['wall_seconds'] for c in cost_by_model.values()),
        'includes':'Inner hyperparameter/epoch search and refit; excludes preprocessing and prediction.',
        'stacking_mode':config['stacking_mode']})
    write_json(out/'bundle.json',{'feature_order':list(FEATURE_ORDER),'base_models':paths,'ensembles':ensemble_paths,
        'thresholds':ensemble_thresholds,'base_thresholds':thresholds,'data_fingerprint':fingerprint(pd.concat([train,val,test])),
        'stacking_mode':config['stacking_mode'],'seed':config['seed']})
    from .inference import Detector
    from .utils import measured
    with measured() as usage:
        detector=Detector(out)
        timings=benchmark(lambda texts:np.array([r['probability'] for r in detector.predict(texts)]),val.text,batch_size=16)
    write_json(out/'evaluation/bundle_benchmark.json',{**usage,**timings,'model_bytes':size_bytes(out/'ensembles')+sum(size_bytes(out/p) for p in paths.values())})
    return results


def family_experiment(config,family):
    train,_,_=read_prepared(config)
    fit,test=family_holdout(train,family,config['seed']);fit,val=group_holdout(fit,5,config['seed']+1)
    assert_isolated([fit,val,test]);out=Path(config['run_dir'])/'family_holdout'/digest(family)[:12]
    write_json(out/'manifest.json',{'family':family,'fit':fit.sample_id.tolist(),'validation':val.sample_id.tolist(),'test':test.sample_id.tolist()})
    model,costs=fit_base('tfidf',fit,config,out/'model')
    return evaluate('tfidf',val,test,predict_base(model,'tfidf',val.text),predict_base(model,'tfidf',test.text),out,costs)
