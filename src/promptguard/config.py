import json
from pathlib import Path


def load_config(path):
    p = Path(path)
    if p.suffix in {'.yaml', '.yml'}:
        import yaml
        config = yaml.safe_load(p.read_text())
    else: config = json.loads(p.read_text())
    if config['stacking_mode'] not in {'oof', 'holdout'}: raise ValueError('Unknown stacking mode')
    if config['oof_folds'] < 2: raise ValueError('At least two OOF folds required')
    return config
