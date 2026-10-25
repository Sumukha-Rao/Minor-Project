"""Load only trusted local model artifacts: joblib/pickle can execute code."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .oof import FEATURE_ORDER


class Detector:
    def __init__(self,run_dir,model='stacking',batch_size=4):
        self.root=Path(run_dir);self.kind=model;self.models={};self.meta=None
        if model in FEATURE_ORDER:
            info=json.loads((self.root/f'{model}_inference.json').read_text())
            paths={model:info['path']};self.threshold=info['threshold'];self.order=(model,)
        else:
            info=json.loads((self.root/'bundle.json').read_text())
            if info['feature_order']!=list(FEATURE_ORDER):raise ValueError('Bundle has incompatible feature order')
            if model not in info['ensembles']:raise ValueError(f'Unknown ensemble {model}')
            self.meta=joblib.load(self._path(info['ensembles'][model]));self.order=tuple(self.meta.order)
            paths={k:info['base_models'][k] for k in self.order};self.threshold=info['thresholds'][model]
        for name,path in paths.items():
            if name=='tfidf':self.models[name]=joblib.load(self._path(path))
            else:
                from .transformer import TransformerPredictor
                self.models[name]=TransformerPredictor(self._path(path),batch_size)
    def _path(self,relative):
        result=(self.root/relative).resolve()
        if not result.is_relative_to(self.root.resolve()):raise ValueError('Model path escapes bundle directory')
        if not result.exists():raise FileNotFoundError(result)
        return result
    def predict(self,texts,individual=True):
        single=isinstance(texts,str);texts=[texts] if single else list(texts)
        if not texts:return []
        if any(not isinstance(t,str) or not t.strip() for t in texts):raise ValueError('Inputs must be nonempty strings')
        columns={}
        for name in self.order:
            model=self.models[name]
            columns[name]=model.predict_proba(texts)[:,1] if name=='tfidf' else model.predict(texts)
        features=pd.DataFrame(columns,columns=self.order)
        probabilities=self.meta.predict(features) if self.meta else features.iloc[:,0].to_numpy()
        result=[]
        for i,p in enumerate(probabilities):
            row={'prediction':'PROMPT_INJECTION' if p>=self.threshold else 'BENIGN','probability':float(p)}
            if individual:row['individual_models']={('tfidf_logistic_regression' if k=='tfidf' else k):float(v[i]) for k,v in columns.items()}
            result.append(row)
        return result[0] if single else result
