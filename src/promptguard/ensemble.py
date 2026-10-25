import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from .oof import FEATURE_ORDER


def ordered_features(frame, order=FEATURE_ORDER):
    if not isinstance(frame,pd.DataFrame) or list(frame.columns)!=list(order):
        raise ValueError(f'Feature order must be exactly {list(order)}')
    values=frame.to_numpy(dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError('Invalid base probabilities')
    return values


class StackingEnsemble:
    def __init__(self,order=FEATURE_ORDER,seed=42):
        self.order=tuple(order);self.model=LogisticRegression(C=1.,max_iter=2000,random_state=seed)
    def fit(self,features,y,*,provenance):
        if provenance not in {'oof','disjoint_holdout'}:raise ValueError('In-sample stacking is forbidden')
        self.provenance=provenance
        self.model.fit(ordered_features(features,self.order),y)
        return self
    def predict(self,features):return self.model.predict_proba(ordered_features(features,self.order))[:,1]


class WeightedEnsemble:
    def __init__(self,order=FEATURE_ORDER):self.order=tuple(order)
    def fit(self,features,y,*,provenance):
        if provenance not in {'oof','disjoint_holdout','validation'}:raise ValueError('Never tune weights on test')
        x=ordered_features(features,self.order);n=x.shape[1]
        result=minimize(lambda w:log_loss(y,np.clip(x@w,1e-7,1-1e-7),labels=[0,1]),
            np.full(n,1/n),method='SLSQP',bounds=[(0.,1.)]*n,
            constraints=[{'type':'eq','fun':lambda w: w.sum()-1}],options={'maxiter':1000,'ftol':1e-10})
        if not result.success:raise RuntimeError(f'Weight optimization failed: {result.message}')
        self.weights=np.maximum(result.x,0);self.weights/=self.weights.sum();self.provenance=provenance
        return self
    def predict(self,features):return ordered_features(features,self.order)@self.weights
