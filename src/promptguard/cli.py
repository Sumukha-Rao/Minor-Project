import argparse
import json
from pathlib import Path
from .config import load_config


def main():
    p=argparse.ArgumentParser(description='Local-data prompt injection detection experiments')
    p.add_argument('--config',default='configs/default.json');sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('inspect');sub.add_parser('prepare');sub.add_parser('preflight');sub.add_parser('run')
    base=sub.add_parser('train-base');base.add_argument('model',choices=['tfidf','distilbert','bert','roberta'])
    oof=sub.add_parser('oof');oof.add_argument('--models',nargs='+',default=['bert','roberta','distilbert','tfidf'],choices=['bert','roberta','distilbert','tfidf'])
    infer=sub.add_parser('predict');infer.add_argument('--run-dir',required=True);infer.add_argument('--model',default='stacking');infer.add_argument('--text');infer.add_argument('--input-jsonl')
    family=sub.add_parser('family-holdout');family.add_argument('family')
    a=p.parse_args();config=load_config(a.config)
    if a.command=='inspect':
        from .data import inventory
        result=inventory(config['dataset_root'],'reports')
    elif a.command=='prepare':
        from .prepare import prepare
        result=prepare(config)
    elif a.command=='predict':
        from .inference import Detector
        d=Detector(a.run_dir,a.model)
        if bool(a.text)==bool(a.input_jsonl):p.error('Provide exactly one of --text or --input-jsonl')
        texts=a.text if a.text else [json.loads(line)['text'] for line in Path(a.input_jsonl).read_text().splitlines() if line.strip()]
        result=d.predict(texts)
    else:
        from .experiment import independent,run_oof,full_experiment,preflight,family_experiment
        if a.command=='train-base':result=independent(config,a.model)
        elif a.command=='oof':result=run_oof(config,a.models).describe().to_dict()
        elif a.command=='preflight':result=preflight(config)
        elif a.command=='family-holdout':result=family_experiment(config,a.family)
        else:result=full_experiment(config)
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()
