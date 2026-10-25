"""Read local data only; explicit schema/label mapping is mandatory for admission."""
import io
import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from .utils import digest, write_json

FORMATS = {'.csv', '.json', '.jsonl', '.parquet', '.xlsx', '.xls'}
TEXT_NAMES = {'text', 'query', 'prompt', 'prompt_injections', 'target_text', 'inject_text'}


def read_table(path):
    p = Path(path)
    if p.suffix == '.csv': return pd.read_csv(p, keep_default_na=False)
    if p.suffix == '.jsonl': return pd.read_json(p, lines=True)
    if p.suffix == '.json': return pd.read_json(p)
    if p.suffix in {'.xlsx', '.xls'}: return pd.read_excel(p)
    if p.suffix == '.parquet':
        try: return pd.read_parquet(p)
        except ImportError:
            cli = shutil.which('duckdb')
            if not cli: raise ImportError('Parquet needs pyarrow or local DuckDB CLI')
            sql = "SELECT * FROM read_parquet('" + str(p.resolve()).replace("'", "''") + "')"
            result = subprocess.run([cli, '-csv', '-c', sql], check=True, capture_output=True, text=True)
            return pd.read_csv(io.StringIO(result.stdout), keep_default_na=False)
    raise ValueError(f'Unsupported file: {path}')


def normalize_label(value, mapping):
    # A value such as 2 or an unknown string must never silently become benign.
    key = str(value)
    if key not in mapping: raise ValueError(f'Unverified label: {value!r}')
    mapped = mapping[key]
    if type(mapped) is not int or mapped not in (0, 1): raise ValueError('Mapping must contain integer 0/1')
    return mapped


def inventory(root, output):
    root = Path(root); entries = []; text_hashes = {}; label_maps = {}; overlaps = []
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in FORMATS or '.git' in p.parts: continue
        item = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': __import__('hashlib').sha256(p.read_bytes()).hexdigest()}
        try:
            df = read_table(p); item.update(rows=len(df), columns={c: str(t) for c,t in df.dtypes.items()},
                missing={c: int((df[c].isna() | df[c].astype(str).str.strip().eq('')).sum()) for c in df},
                duplicate_rows=int(df.astype(str).duplicated().sum()))
            item['labels'] = {c: df[c].astype(str).value_counts().to_dict() for c in df if 'label' in c or c == 'subcategory'}
            texts = [c for c in df if c in TEXT_NAMES or c.endswith('_attack')]
            item['text_columns'] = {}
            for c in texts:
                s = df[c].dropna().astype(str); lengths = s.str.len()
                item['text_columns'][c] = {'exact_duplicates': int(s.duplicated().sum()),
                    'length_chars': lengths.quantile([0,.5,.9,.95,.99,1]).to_dict(),
                    'length_words': s.str.split().str.len().quantile([.5,.95,1]).to_dict()}
            item['split_hint'] = 'test' if p.name.startswith('test') else 'train' if p.name.startswith('train') else 'unspecified'
            if len(texts) == 1:
                col = texts[0]; hs = set(df[col].astype(str).map(digest))
                for prev, old in text_hashes.items():
                    n = len(hs & old)
                    if n:
                        overlap={'left': prev, 'right': str(p), 'shared_exact_texts': n, 'identical_text_sets': hs == old}
                        if 'label' in df and prev in label_maps and Path(prev).relative_to(root).parts[0] == p.relative_to(root).parts[0]:
                            current=dict(zip(df[col].astype(str).map(digest),df['label'].astype(str)))
                            overlap['same_collection_label_conflicts']=sum(current[h]!=label_maps[prev][h] for h in hs & old)
                        overlaps.append(overlap)
                text_hashes[str(p)] = hs
                label = next((c for c in ['label'] if c in df), None)
                if label:
                    label_maps[str(p)]=dict(zip(df[col].astype(str).map(digest),df[label].astype(str)))
                    item['exact_text_label_conflicts'] = int((df.groupby(col)[label].nunique()>1).sum())
                    item['label_examples'] = {str(k): g[col].head(3).tolist() for k,g in df.groupby(label)}
        except Exception as e: item['error'] = f'{type(e).__name__}: {e}'
        entries.append(item)
    report = {'files': entries, 'exact_cross_file_overlap': overlaps,
              'note': 'Counts include mirrored formats; do not sum them as independent samples. Examples are untrusted dataset text.'}
    write_json(Path(output)/'inventory.json', report)
    return report


def load_verified(config):
    rows = []; quarantine = []
    for spec in config['sources']:
        if not spec.get('evidence'): raise ValueError('Every source needs label evidence')
        files = sorted(Path(config['dataset_root']).glob(spec['glob']))
        if not files: raise FileNotFoundError(spec['glob'])
        for p in files:
            df = read_table(p)
            for i, row in df.iterrows():
                raw = row[spec['text']]
                if not isinstance(raw, str) or not raw.strip():
                    quarantine.append({'source':str(p),'row':int(i),'reason':'empty/non-string text'});continue
                original = str(row[spec['label']]) if 'label' in spec else 'constant'
                try:
                    y = normalize_label(original, spec['mapping']) if 'label' in spec else spec['constant_label']
                    if y not in (0,1): raise ValueError('Invalid constant label')
                except ValueError:
                    quarantine.append({'source':str(p),'row':int(i),'label':original,'reason':'unverified label'});continue
                rows.append({'sample_id':digest(raw),'text':raw,'label':y,'original_label':original,
                    'source':str(p), 'source_row':int(i), 'audit_only':spec.get('audit_only',False), 'category':str(row[spec['category']]) if 'category' in spec else '',
                    'original_split':spec.get('split', 'test' if p.name.startswith('test') else 'train' if p.name.startswith('train') else 'validation' if p.name.startswith(('validation','val-')) else 'unspecified'), 'template_numbers':spec.get('template_numbers',False),
                    'metadata':json.dumps(row.to_dict(),ensure_ascii=False,default=str)})
    return pd.DataFrame(rows), pd.DataFrame(quarantine)
