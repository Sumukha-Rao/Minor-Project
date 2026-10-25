"""Subsample only already isolated partitions. Does not invent or download data."""
import copy
from pathlib import Path
import pandas as pd
from promptguard.config import load_config
from promptguard.splitting import assert_isolated
from promptguard.utils import fingerprint, write_json

config=load_config('configs/default.json')
frame=pd.read_json(Path(config['prepared_dir'])/'samples.jsonl',lines=True)
parts=[]
for name,df in frame.groupby('split'):
    chosen=[]
    for label,g in df.groupby('label'):
        # One representative per group emphasizes group diversity over templates.
        chosen.append(g.sort_values('sample_id').drop_duplicates('group').sample(
            n=min(64,len(g.group.unique())),random_state=config['seed']))
    parts.append(pd.concat(chosen).sort_values('sample_id'))
assert_isolated(parts)
small=pd.concat(parts,ignore_index=True)
out=Path('data/smoke');out.mkdir(parents=True,exist_ok=True)
small.to_json(out/'samples.jsonl',orient='records',lines=True,force_ascii=False)
small[['sample_id','group','split','label','source']].to_csv(out/'split_manifest.csv',index=False)
write_json(out/'preparation.json',{'fingerprint':fingerprint(small),'purpose':'Real local-data wiring smoke test, not a research result','rows':len(small)})
config.update(prepared_dir=str(out),run_dir='runs/smoke',oof_folds=2)
config['tfidf'].update(search='smoke',cv_folds=2)
config['transformer'].update(epochs=1,max_length=128,batch_size=2)
write_json('configs/smoke.json',config)
print(small.groupby(['split','label']).size())
