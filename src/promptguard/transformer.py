"""Fresh pretrained initialization per fold; inner-only epoch selection."""
import gc
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from .splitting import group_holdout
from .utils import seed_everything, write_json, fingerprint, digest, measured


def device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_base(identifier, cfg):
    args={'local_files_only':cfg.get('local_files_only',True),'trust_remote_code':False}
    if cfg.get('cache_dir'): args['cache_dir']=cfg['cache_dir']
    tokenizer=AutoTokenizer.from_pretrained(identifier,**args)
    model=AutoModelForSequenceClassification.from_pretrained(identifier,num_labels=2,
        id2label={0:'BENIGN',1:'PROMPT_INJECTION'},label2id={'BENIGN':0,'PROMPT_INJECTION':1},**args)
    return tokenizer,model


def make_loader(frame, tokenizer, cfg, shuffle=False):
    rows=list(zip(frame.text.tolist(),frame.label.astype(int).tolist()))
    def collate(batch):
        texts,labels=zip(*batch)
        result=tokenizer(list(texts),padding=True,truncation=True,max_length=cfg['max_length'],return_tensors='pt')
        result['labels']=torch.tensor(labels,dtype=torch.long)
        return result
    return DataLoader(rows,batch_size=cfg['batch_size'],shuffle=shuffle,collate_fn=collate,num_workers=0)


class TransformerPredictor:
    def __init__(self, path, batch_size=4, max_length=None):
        self.path=Path(path);self.device=device();self.batch_size=batch_size
        settings=json.loads((self.path/'inference.json').read_text())
        self.max_length=max_length or settings['max_length']
        torch.set_num_threads(settings.get('num_threads',4))
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(self.path,local_files_only=True,trust_remote_code=False).to(self.device).eval()
    @torch.inference_mode()
    def predict(self,texts):
        texts=list(texts);scores=[]
        for i in range(0,len(texts),self.batch_size):
            batch=self.tokenizer(texts[i:i+self.batch_size],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt').to(self.device)
            scores.extend(self.model(**batch).logits.softmax(-1)[:,1].cpu().tolist())
        return np.asarray(scores)
    def truncation_stats(self,texts):
        lengths=[len(self.tokenizer(t,truncation=False)['input_ids']) for t in texts]
        return {'n':len(lengths),'truncated_fraction':float(np.mean(np.asarray(lengths)>self.max_length)),
                'max_length':self.max_length,'token_length_quantiles':np.quantile(lengths,[.5,.95,1]).tolist()}


def _fit(train, validation, identifier, cfg, output, seed, epochs, early_stop):
    """Epoch-boundary resume, with data/config identity checks and complete RNG state."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed_everything(seed);torch.set_num_threads(cfg.get('num_threads',4))
    tokenizer,model=load_base(identifier,cfg);dev=device();model.to(dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    loader=make_loader(train,tokenizer,cfg,shuffle=True)
    accumulation=cfg['gradient_accumulation']
    total_steps=math.ceil(len(loader)/accumulation)*epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total_steps*cfg['warmup_ratio']),total_steps)
    amp=bool(cfg['mixed_precision'] and dev.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    counts=np.bincount(train.label.astype(int),minlength=2)
    weighting=cfg.get('class_weight','auto')
    weights=None
    if weighting=='balanced' or (weighting=='auto' and counts.max()/counts.min()>1.5):
        weights=torch.tensor(len(train)/(2*counts),dtype=torch.float32,device=dev)
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    signature=digest(json.dumps({'train':fingerprint(train),'validation':fingerprint(validation) if validation is not None else None,
        'identifier':identifier,'config':cfg,'epochs':epochs,'seed':seed,'early_stop':early_stop},sort_keys=True))
    resume=output/'resume.pt';start=0;best=-1.;bad=0;best_epoch=0;history=[]
    if cfg.get('resume') and resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False) # trusted locally generated checkpoint only
        if state['signature']!=signature:raise ValueError(f'Stale checkpoint: {output}; use a new run directory')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        start=state['epoch']+1;best=state['best'];bad=state['bad'];best_epoch=state['best_epoch'];history=state['history']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if dev.type=='cuda' and state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    if dev.type=='cuda':torch.cuda.reset_peak_memory_stats()
    with measured() as costs:
        for epoch in range(start,epochs):
            if early_stop and bad>=cfg['patience']:break
            model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;n=0
            for step,batch in enumerate(loader):
                labels=batch.pop('labels').to(dev);batch={k:v.to(dev) for k,v in batch.items()}
                # Correct scale even for a final partial accumulation window.
                group_start=(step//accumulation)*accumulation
                group_size=min(accumulation,len(loader)-group_start)
                with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=amp):
                    logits=model(**batch).logits;raw_loss=criterion(logits,labels);loss=raw_loss/group_size
                scaler.scale(loss).backward();loss_sum+=raw_loss.item()*len(labels);n+=len(labels)
                if (step+1)%accumulation==0 or step+1==len(loader):
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'])
                    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=old_scale:scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            entry={'epoch':epoch+1,'train_loss':loss_sum/n,'learning_rate':scheduler.get_last_lr()[0]}
            if validation is not None:
                model.eval();val_loss=0.;vn=0;ys=[];pred=[]
                with torch.inference_mode():
                    for batch in make_loader(validation,tokenizer,cfg):
                        labels=batch.pop('labels').to(dev);batch={k:v.to(dev) for k,v in batch.items()}
                        logits=model(**batch).logits
                        val_loss+=criterion(logits,labels).item()*len(labels);vn+=len(labels)
                        ys.extend(labels.cpu().tolist());pred.extend(logits.argmax(-1).cpu().tolist())
                score=f1_score(ys,pred,average='macro',zero_division=0)
                entry.update(validation_loss=val_loss/vn,validation_macro_f1=score)
            else:score=float(epoch)
            if score>best:
                best=score;bad=0;best_epoch=epoch+1
                model.save_pretrained(output/'best',safe_serialization=True);tokenizer.save_pretrained(output/'best')
                write_json(output/'best/inference.json',{'max_length':cfg['max_length'],'num_threads':cfg.get('num_threads',4)})
            else:bad+=1
            history.append(entry);write_json(output/'history.json',history)
            state={'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'bad':bad,
                'best_epoch':best_epoch,'history':history,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if dev.type=='cuda' else None}
            torch.save(state,output/'resume.tmp');(output/'resume.tmp').replace(resume)
            print(f'{output}: {entry}',flush=True)
    if dev.type=='cuda':costs['gpu_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    write_json(output/'training.json',{'best_epoch':best_epoch,'signature':signature,'costs':costs})
    del model,optimizer,loader;gc.collect()
    if dev.type=='cuda':torch.cuda.empty_cache()
    return best_epoch,costs


def train_transformer(train,name,config,output):
    cfg=config['transformer'];identifier=cfg['models'][name];output=Path(output)
    # Neither outer OOF holdout nor final validation/test is used for epoch selection.
    fit,select=group_holdout(train,min(5,config['oof_folds']),config['seed']+11)
    best,c1=_fit(fit,select,identifier,cfg,output/'selection',config['seed'],cfg['epochs'],True)
    _,c2=_fit(train,None,identifier,cfg,output/'refit',config['seed'],best,False)
    costs={'wall_seconds':c1['wall_seconds']+c2['wall_seconds'],
           'process_peak_rss_mib':max(c1['process_peak_rss_mib'],c2['process_peak_rss_mib']),
           'memory_note':c2['memory_note'],'selected_epochs':best}
    if 'gpu_peak_allocated_mib' in c2:costs['gpu_peak_allocated_mib']=max(c1['gpu_peak_allocated_mib'],c2['gpu_peak_allocated_mib'])
    write_json(output/'training.json',costs)
    if not cfg.get('retain_optimizer_checkpoints', False):
        # Complete models and logs remain cached; optimizer state is only for interrupted runs.
        for state in output.glob('*/resume.pt'): state.unlink()
    return TransformerPredictor(output/'refit/best',cfg['batch_size']),costs


def loss_plot(output):
    import plotly.graph_objects as go
    output=Path(output);fig=go.Figure()
    for path in output.rglob('history.json'):
        history=json.loads(path.read_text())
        for key in ['train_loss','validation_loss']:
            values=[h[key] for h in history if key in h]
            if values:fig.add_trace(go.Scatter(x=list(range(1,len(values)+1)),y=values,name=str(path.parent.relative_to(output))+'/'+key))
    fig.update_layout(title='Training and inner-validation loss',xaxis_title='Epoch',yaxis_title='Cross entropy')
    fig.write_html(output/'loss_curves.html',include_plotlyjs=True)
