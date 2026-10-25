"""Only dataset-provided categories; no keyword-created attack labels."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from .splitting import group_holdout, assert_isolated
from .utils import write_json


def family_holdout(frame,family,seed=42):
    if 'category' not in frame:raise ValueError('No category metadata')
    attack=frame[(frame.label==1)&frame.category.eq(family)]
    if attack.empty:raise ValueError(f'No labeled positive attack family: {family}')
    reserved=set(attack.group)
    remaining=frame[~frame.group.isin(reserved)].reset_index(drop=True)
    fit,hold=group_holdout(remaining,5,seed)
    benign=hold[hold.label==0]
    # The unused positive holdout may safely return to fit, by whole group only.
    benign_groups=set(benign.group)
    train=remaining[~remaining.group.isin(benign_groups)].reset_index(drop=True)
    evaluation=pd.concat([attack,benign],ignore_index=True)
    assert_isolated([train,evaluation])
    return train,evaluation


def category_availability(frame,output):
    positive=frame[(frame.label==1)&frame.category.fillna('').ne('')]
    report={'available_positive_families':positive.category.value_counts().to_dict(),
        'family_held_out_status':'available; run explicit family experiment' if len(positive) else 'not supported by admitted data',
        'unavailable_unless_annotated':['instruction_override','role_spoofing','indirect','prompt_extraction',
            'encoded_or_obfuscated','multilingual','benign_security_quotations','long_context','semantically_subtle'],
        'note':'No automatic category labels or claims of robustness coverage.'}
    write_json(Path(output)/'robustness_availability.json',report);return report


def disagreement(frame,probabilities,thresholds,output):
    result=frame[['sample_id','label','source','category','text']].copy()
    decisions=[]
    for name in probabilities:
        result[name+'_probability']=probabilities[name].to_numpy()
        decision=(probabilities[name].to_numpy()>=thresholds[name]).astype(int)
        result[name+'_prediction']=decision;decisions.append(decision)
    votes=np.stack(decisions,axis=1)
    result['disagreement']=votes.min(axis=1)!=votes.max(axis=1)
    result['probability_range']=probabilities.max(axis=1).to_numpy()-probabilities.min(axis=1).to_numpy()
    result.to_csv(Path(output)/'prediction_comparison.csv',index=False)
    result[result.disagreement].to_csv(Path(output)/'disagreements.csv',index=False)
    write_json(Path(output)/'disagreement_summary.json',{'n':len(result),'disagreement_rate':float(result.disagreement.mean())})


def paired_cluster_bootstrap(frame,predictions,reference='tfidf',repeats=500,seed=42):
    rng=np.random.default_rng(seed);groups=frame.groupby('group').indices;keys=list(groups);deltas={k:[] for k in predictions if k!=reference}
    y=frame.label.to_numpy()
    for _ in range(repeats):
        idx=np.concatenate([groups[keys[i]] for i in rng.integers(0,len(keys),len(keys))])
        ref=f1_score(y[idx],np.asarray(predictions[reference])[idx],average='macro',labels=[0,1],zero_division=0)
        for name in deltas:
            deltas[name].append(f1_score(y[idx],np.asarray(predictions[name])[idx],average='macro',labels=[0,1],zero_division=0)-ref)
    return {'reference':reference,'resampling_unit':'duplicate group','repeats':repeats,
        'note':'Exploratory paired intervals; not corrected for multiple comparisons or source sampling bias.',
        'macro_f1_delta_95ci':{k:np.quantile(v,[.025,.5,.975]).tolist() for k,v in deltas.items()}}
