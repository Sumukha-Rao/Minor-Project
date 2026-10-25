"""OOF prediction store with explicit fit/holdout identities and cache guards."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .splitting import group_folds, assert_isolated
from .utils import fingerprint, digest, write_json
from .evaluation import check_probabilities

FEATURE_ORDER=('bert','roberta','distilbert','tfidf')


def code_fingerprint():
    return digest(''.join(p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))))


def generate_oof(train,config,output,fit_predict,model_names=FEATURE_ORDER):
    """fit_predict(name, fold_train, fold_holdout, path) MUST fit afresh.

    Implementations receive no validation/test rows. Hyperparameter/epoch selection
    belongs inside fold_train. The callback is dependency-injected for auditing tests.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    folds=group_folds(train,config['oof_folds'],config['seed'])
    matrix=np.full((len(train),len(model_names)),np.nan);seen=np.zeros_like(matrix,dtype=int)
    signature=digest(json.dumps({'data':fingerprint(train),'config':config,'code':code_fingerprint(),
                                'models':list(model_names)},sort_keys=True))
    manifest=[]
    for fold,(a,b) in enumerate(folds):
        fit=train.iloc[a].reset_index(drop=True);held=train.iloc[b].reset_index(drop=True)
        assert_isolated([fit,held])
        manifest.append({'fold':fold,'fit_ids':fit.sample_id.tolist(),'holdout_ids':held.sample_id.tolist(),
                         'fit_groups':sorted(fit.group.unique()),'holdout_groups':sorted(held.group.unique())})
        for col,name in enumerate(model_names):
            path=output/f'fold_{fold}'/name;path.mkdir(parents=True,exist_ok=True)
            cached=path/'predictions.npz';metadata=path/'cache.json'
            if cached.exists() and metadata.exists():
                meta=json.loads(metadata.read_text())
                if meta['signature']!=signature or meta['holdout_ids']!=held.sample_id.tolist():
                    raise ValueError(f'Stale OOF cache at {path}; choose a new run directory')
                with np.load(cached,allow_pickle=False) as z:p=z['p']
            else:
                p=check_probabilities(fit_predict(name,fit,held,path),len(held))
                np.savez_compressed(cached,p=p)
                write_json(metadata,{'signature':signature,'holdout_ids':held.sample_id.tolist(),
                                     'fit_ids':fit.sample_id.tolist(),'model':name})
            matrix[b,col]=check_probabilities(p,len(b));seen[b,col]+=1
        write_json(output/'fold_manifest.json',manifest)
    if not np.all(seen==1) or not np.isfinite(matrix).all():raise ValueError('OOF coverage is not exactly once per model/sample')
    result=pd.DataFrame(matrix,columns=model_names,index=train.sample_id.tolist())
    result.index.name='sample_id';result.to_csv(output/'oof_probabilities.csv')
    write_json(output/'metadata.json',{'signature':signature,'feature_order':list(model_names),'rows':len(train)})
    return result
