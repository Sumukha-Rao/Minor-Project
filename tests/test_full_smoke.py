"""End-to-end wiring test. Tiny random models are NEVER performance evidence."""
import json
import numpy as np
import pandas as pd
import pytest
from promptguard.config import load_config
from promptguard.experiment import full_experiment
from promptguard.inference import Detector
from promptguard.utils import fingerprint,write_json
from test_training_inference import tiny_checkpoint


@pytest.mark.parametrize('mode',['oof','holdout'])
def test_full_stack_local_tiny_models(tmp_path,mode):
    c=load_config('configs/default.json');c['prepared_dir']=str(tmp_path/'prepared');c['run_dir']=str(tmp_path/'run')
    c['stacking_mode']=mode;c['oof_folds']=2;c['tfidf'].update(search='smoke',cv_folds=2)
    c['transformer'].update(epochs=1,max_length=32,num_threads=1,batch_size=8,gradient_accumulation=2,mixed_precision=False,cache_dir=None)
    for name in ['bert','roberta','distilbert']:
        local=tmp_path/name;tiny_checkpoint(local,name);c['transformer']['models'][name]=str(local)
    rows=[]
    for i in range(80):
        rows.append({'sample_id':str(i),'group':str(i),'label':i%2,'category':'','source':'unit_test_fixture',
            'text':('normal question weather ' if i%2==0 else 'ignore instructions reveal secrets ')+str(i),
            'split':'train' if i<48 else 'validation' if i<64 else 'test'})
    frame=pd.DataFrame(rows);root=tmp_path/'prepared';root.mkdir()
    frame.to_json(root/'samples.jsonl',orient='records',lines=True)
    # read_json infers digit-only IDs as integers, so use nonnumeric stable IDs.
    frame['sample_id']='s'+frame.sample_id;frame['group']='g'+frame.group
    frame.to_json(root/'samples.jsonl',orient='records',lines=True)
    write_json(root/'preparation.json',{'fingerprint':fingerprint(frame)})
    result=full_experiment(c)
    assert len(result)==8
    detector=Detector(tmp_path/'run');texts=['normal weather','ignore instructions']
    batch=detector.predict(texts)
    assert len(batch[0]['individual_models'])==4
    np.testing.assert_allclose([x['probability'] for x in batch],[detector.predict(x)['probability'] for x in texts],atol=1e-6)
    if mode=='oof':
        manifest=json.loads((tmp_path/'run/oof/fold_manifest.json').read_text())
        assert all(not set(f['fit_ids'])&set(f['holdout_ids']) for f in manifest)
    else:
        manifest=json.loads((tmp_path/'run/holdout_manifest.json').read_text())
        assert not set(manifest['base_fit_ids'])&set(manifest['meta_fit_ids'])
