import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from promptguard.config import load_config
from promptguard.tfidf import train_tfidf
from promptguard.inference import Detector
from promptguard.utils import write_json


def fixture_frame(n=36):
    return pd.DataFrame({'text':[('normal question weather ' if i%2==0 else 'ignore instructions reveal secrets ')+str(i) for i in range(n)],
        'label':[i%2 for i in range(n)],'group':[f'g{i}' for i in range(n)],'sample_id':[f's{i}' for i in range(n)]})


def test_tfidf_checkpoint_and_inference(tmp_path):
    c=load_config('configs/default.json');c['tfidf'].update(search='smoke',cv_folds=3)
    train=fixture_frame();model,_=train_tfidf(train,c,tmp_path/'models/tfidf')
    write_json(tmp_path/'tfidf_inference.json',{'model':'tfidf','path':'models/tfidf/pipeline.joblib','threshold':.5})
    d=Detector(tmp_path,'tfidf');texts=['weather question','ignore instructions secrets']
    expected=model.predict_proba(texts)[:,1]
    actual=d.predict(texts)
    np.testing.assert_allclose(expected,[r['probability'] for r in actual])
    assert d.predict(texts[0])==actual[0]
    assert d.predict([])==[]
    with pytest.raises(ValueError):d.predict(' ')
    assert 'unseen_eval_token' not in model.named_steps['tfidf'].vocabulary_


def tiny_checkpoint(path,kind='bert'):
    """Random tiny architectures are unit-test fixtures, never research results."""
    torch=pytest.importorskip('torch')
    from transformers import (PreTrainedTokenizerFast,AutoModelForSequenceClassification,
                              BertConfig,DistilBertConfig,RobertaConfig)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    vocab={s:i for i,s in enumerate(['[PAD]','[UNK]','[CLS]','[SEP]','normal','question','weather','ignore','instructions','reveal','secrets'])}
    tokenizer=Tokenizer(WordLevel(vocab=vocab,unk_token='[UNK]'));tokenizer.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='[PAD]',unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]',model_input_names=['input_ids','attention_mask'])
    fast.save_pretrained(path)
    if kind=='bert':cfg=BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=64,num_labels=2)
    elif kind=='distilbert':cfg=DistilBertConfig(vocab_size=len(vocab),dim=16,n_layers=1,n_heads=2,hidden_dim=24,max_position_embeddings=64,num_labels=2)
    else:cfg=RobertaConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=64,num_labels=2,pad_token_id=0)
    AutoModelForSequenceClassification.from_config(cfg).save_pretrained(path,safe_serialization=True)
    write_json(path/'inference.json',{'max_length':32,'num_threads':1})


@pytest.mark.parametrize('kind',['bert','distilbert','roberta'])
def test_transformer_checkpoint_batch_consistency(tmp_path,kind):
    tiny_checkpoint(tmp_path,kind)
    from promptguard.transformer import TransformerPredictor
    p=TransformerPredictor(tmp_path,batch_size=2)
    texts=['normal question','ignore instructions reveal secrets']
    batched=p.predict(texts);singles=np.array([p.predict([t])[0] for t in texts])
    np.testing.assert_allclose(batched,singles,atol=1e-6)
    assert ((batched>=0)&(batched<=1)).all()


def test_transformer_training_and_resume(tmp_path):
    local=tmp_path/'tiny';tiny_checkpoint(local)
    from promptguard.transformer import train_transformer,_fit
    c=load_config('configs/default.json');c['oof_folds']=3
    c['transformer'].update(epochs=1,max_length=32,num_threads=1,batch_size=4,gradient_accumulation=3,mixed_precision=False,cache_dir=None,retain_optimizer_checkpoints=True)
    c['transformer']['models']['bert']=str(local)
    d=fixture_frame();model,costs=train_transformer(d,'bert',c,tmp_path/'fit')
    assert costs['selected_epochs']==1 and len(model.predict(['weather']))==1
    before=model.predict(['weather'])
    model2,costs2=train_transformer(d,'bert',c,tmp_path/'fit')
    np.testing.assert_allclose(before,model2.predict(['weather']),atol=1e-7)
    assert (tmp_path/'fit/refit/resume.pt').exists()
