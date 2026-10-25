"""Validation-only thresholds; explicit undefined metrics for single-class slices."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc,
    classification_report, brier_score_loss, log_loss)
from sklearn.calibration import calibration_curve
from .utils import write_json


def check_probabilities(p, n=None):
    p=np.asarray(p,dtype=float)
    if p.ndim!=1 or (n is not None and len(p)!=n) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Expected finite positive-class probabilities in [0,1]')
    return p


def choose_threshold(y,p):
    p=check_probabilities(p,len(y));y=np.asarray(y)
    if len(np.unique(y))!=2: raise ValueError('Threshold selection requires both validation classes')
    # Exact candidates at every distinct validation prediction, including all-negative.
    order=np.argsort(-p,kind='stable');scores=p[order];labels=y[order]
    ends=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1]
    tp=np.cumsum(labels)[ends]; fp=(ends+1)-tp; fn=y.sum()-tp;tn=len(y)-y.sum()-fp
    pos=2*tp/np.maximum(2*tp+fp+fn,1);neg=2*tn/np.maximum(2*tn+fp+fn,1)
    thresholds=np.r_[np.nextafter(scores[0],np.inf),scores[ends]]
    allneg=(len(y)-y.sum())/(2*len(y)-y.sum())
    metrics=np.r_[allneg,(pos+neg)/2]
    best=np.flatnonzero(np.isclose(metrics,metrics.max(),rtol=0,atol=1e-12))
    return float(thresholds[min(best,key=lambda i:abs(thresholds[i]-.5))])


def metrics(y,p,threshold=.5):
    y=np.asarray(y,dtype=int);p=check_probabilities(p,len(y));pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    bins=np.minimum((p*10).astype(int),9);ece=0.
    for b in range(10):
        mask=bins==b
        if mask.any(): ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    both=len(np.unique(y))==2
    precision,recall,_=precision_recall_curve(y,p) if both else (None,None,None)
    return {'n':len(y),'threshold':threshold,'accuracy':accuracy_score(y,pred),
        'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),
        'f1':f1_score(y,pred,zero_division=0),'macro_f1':f1_score(y,pred,labels=[0,1],average='macro',zero_division=0),
        'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]],
        'false_positive_rate':float(fp/(fp+tn)) if fp+tn else None,
        'false_negative_rate':float(fn/(fn+tp)) if fn+tp else None,
        'roc_auc':roc_auc_score(y,p) if both else None,
        'pr_auc_trapezoid':auc(recall,precision) if both else None,
        'average_precision':average_precision_score(y,p) if both else None,
        'brier_score':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1]),'ece_10_equal_width':ece,
        'per_class':classification_report(y,pred,labels=[0,1],target_names=['BENIGN','PROMPT_INJECTION'],output_dict=True,zero_division=0)}


def plots(y,p,threshold,output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    y=np.asarray(y);p=np.asarray(p)
    fig=make_subplots(rows=2,cols=2,subplot_titles=['Confusion matrix','ROC','Precision–recall','Calibration'])
    cm=confusion_matrix(y,p>=threshold,labels=[0,1])
    fig.add_trace(go.Heatmap(z=cm,x=['Predicted benign','Predicted injection'],y=['True benign','True injection'],text=cm,texttemplate='%{text}',showscale=False),row=1,col=1)
    if len(np.unique(y))==2:
        fpr,tpr,_=roc_curve(y,p);pr,re,_=precision_recall_curve(y,p)
        fig.add_trace(go.Scatter(x=fpr,y=tpr,name='ROC'),row=1,col=2)
        fig.add_trace(go.Scatter(x=re,y=pr,name='PR'),row=2,col=1)
    observed,mean=calibration_curve(y,p,n_bins=10,strategy='uniform')
    fig.add_trace(go.Scatter(x=mean,y=observed,name='Observed calibration'),row=2,col=2)
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Perfect calibration',line={'dash':'dash'}),row=2,col=2)
    fig.update_layout(height=850,title='Held-out evaluation; threshold fixed on validation')
    fig.write_html(output/'evaluation.html',include_plotlyjs=True)


def benchmark(predict,texts,batch_size=16,repeats=3):
    texts=list(texts)[:128]
    if not texts:return {}
    predict(texts[:min(len(texts),batch_size)])
    singles=[];batches=[]
    for _ in range(repeats):
        for t in texts[:min(16,len(texts))]:
            start=time.perf_counter();predict([t]);singles.append((time.perf_counter()-start)*1000)
        start=time.perf_counter()
        for i in range(0,len(texts),batch_size):predict(texts[i:i+batch_size])
        batches.append((time.perf_counter()-start)*1000/len(texts))
    return {'single_ms_p50':float(np.median(singles)),'single_ms_p95':float(np.percentile(singles,95)),
            'batch_ms_per_sample_mean':float(np.mean(batches)),'batch_size':batch_size,'n':len(texts),
            'includes':'tokenization/vectorization + prediction; warm model; no disk loading'}


def evaluate(name,validation,test,val_p,test_p,output,costs=None):
    output=Path(output)/name;output.mkdir(parents=True,exist_ok=True)
    threshold=choose_threshold(validation.label,val_p)
    result={'model':name,'validation':metrics(validation.label,val_p,threshold),
            'test':metrics(test.label,test_p,threshold),'costs':costs or {}}
    write_json(output/'metrics.json',result)
    write_json(output/'threshold.json',{'threshold':threshold,'selected_on':'validation','objective':'macro_f1'})
    df=test[['sample_id','group','source','category','text','label']].copy()
    df['probability']=test_p;df['prediction']=(np.asarray(test_p)>=threshold).astype(int)
    df.to_csv(output/'predictions.csv',index=False)
    df[df.label!=df.prediction].to_csv(output/'errors.csv',index=False)
    plots(test.label,test_p,threshold,output)
    slices={}
    for col in ['source','category']:
        slices[col]={str(v):metrics(g.label,g.probability,threshold) for v,g in df.groupby(col) if str(v)}
    write_json(output/'slices.json',slices)
    return result


def comparison(results,output):
    import plotly.graph_objects as go
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame([{'model':r['model'],**{k:v for k,v in r['test'].items() if isinstance(v,(int,float))},
                      **{k:v for k,v in r.get('costs',{}).items() if isinstance(v,(int,float))}} for r in results])
    df.to_csv(output/'comparison.csv',index=False)
    fig=go.Figure()
    for m in ['macro_f1','f1','roc_auc','average_precision']:
        fig.add_trace(go.Bar(name=m,x=df.model,y=df[m]))
    fig.update_layout(barmode='group',title='Identical test partition; no ensemble superiority assumed')
    fig.write_html(output/'comparison.html',include_plotlyjs=True)
