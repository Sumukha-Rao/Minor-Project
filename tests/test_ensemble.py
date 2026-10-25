import numpy as np
import pandas as pd
import pytest
from promptguard.oof import generate_oof,FEATURE_ORDER
from promptguard.ensemble import StackingEnsemble,WeightedEnsemble
from promptguard.evaluation import metrics,choose_threshold


def test_oof_fit_identity_and_cache(tmp_path):
    frame=pd.DataFrame({'text':[f'example {i}' for i in range(60)],'label':[i%2 for i in range(60)],
        'group':[str(i) for i in range(60)],'sample_id':[str(i) for i in range(60)]})
    calls=[]
    def train(name,fit,held,path):
        assert not set(fit.sample_id)&set(held.sample_id)
        assert len(fit)==40
        calls.append((name,len(held)))
        return np.full(len(held),.4)
    config={'oof_folds':3,'seed':42}
    result=generate_oof(frame,config,tmp_path,train)
    assert result.shape==(60,4) and len(calls)==12 and not result.isna().any().any()
    generate_oof(frame,config,tmp_path,lambda *a:pytest.fail('cached folds should not retrain'))
    with pytest.raises(ValueError):generate_oof(frame.assign(text='changed'),config,tmp_path,train)


def test_meta_order_provenance_and_weights():
    y=np.array([0,1]*10);x=pd.DataFrame({k:.1+.8*y for k in FEATURE_ORDER})
    meta=StackingEnsemble().fit(x,y,provenance='oof')
    assert (meta.predict(x)[y==1]>.5).all()
    with pytest.raises(ValueError):meta.predict(x[list(reversed(FEATURE_ORDER))])
    with pytest.raises(ValueError):StackingEnsemble().fit(x,y,provenance='in_sample')
    weighted=WeightedEnsemble().fit(x,y,provenance='oof')
    assert np.isclose(weighted.weights.sum(),1) and (weighted.weights>=0).all()
    with pytest.raises(ValueError):WeightedEnsemble().fit(x,y,provenance='test')


def test_threshold_and_single_class_metrics():
    y=np.array([0,0,1,1]);p=np.array([.1,.2,.3,.4]);t=choose_threshold(y,p)
    assert metrics(y,p,t)['macro_f1']==1
    assert metrics([0,0],[.1,.3])['roc_auc'] is None
    assert metrics([0,0],[.1,.3])['false_negative_rate'] is None
    with pytest.raises(ValueError):metrics(y,[np.nan]*4)
