"""Deterministic duplicate components and group-isolated partitions."""
import math
import re
import unicodedata
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .utils import digest


class UnionFind:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]; i = self.parent[i]
        return i
    def union(self, a, b):
        a,b = self.find(a),self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)


def canonical(text):
    # ONLY for grouping. Original model input is preserved byte-for-byte as read.
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def shingles(text, width=5):
    return {text[i:i+width] for i in range(max(1,len(text)-width+1))}


def near_pairs(texts, threshold=.85):
    """Exact Jaccard similarity join via globally ordered prefix filtering.

    Unlike approximate LSH, every pair above the defined threshold is verified.
    This is lexical similarity, not a guarantee of semantic duplicate detection.
    """
    if not 0 < threshold <= 1: raise ValueError('Jaccard threshold must be in (0,1]')
    sets = [shingles(canonical(s)) for s in texts]
    frequency = Counter(t for s in sets for t in s)
    index = defaultdict(list)
    for i in sorted(range(len(sets)), key=lambda j: (len(sets[j]), j)):
        s = sets[i]; n = len(s); minimum = math.ceil(threshold*n - 1e-10)
        prefix = sorted(s, key=lambda t:(frequency[t],t))[:n-minimum+1]
        candidates = set()
        for t in prefix:
            candidates.update(j for j in index[t] if len(sets[j]) >= minimum)
        for j in sorted(candidates):
            other = sets[j]; intersection = len(s & other)
            score = intersection / (n + len(other) - intersection)
            if score + 1e-12 >= threshold: yield j,i,score
        for t in prefix: index[t].append(i)


def deduplicate_and_group(df, threshold=.85):
    df = df.copy(); df['sample_id'] = df.text.map(digest)
    conflicts = set(df.groupby('sample_id').label.nunique().loc[lambda x:x>1].index)
    quarantine = df[df.sample_id.isin(conflicts)].copy()
    df = df[~df.sample_id.isin(conflicts)].copy()
    audit_count=int(df['audit_only'].sum()) if 'audit_only' in df else 0
    # Preserve all origins and categories, including aliases and split conflicts.
    provenance = df.groupby('sample_id').apply(lambda x:x[['source','source_row','category','original_split','metadata']].to_dict('records'), include_groups=False).to_dict()
    original_splits = df.groupby('sample_id').original_split.agg(set).to_dict()
    if 'audit_only' in df: df=df[~df.audit_only].copy()
    duplicates = int(df.duplicated('sample_id').sum())
    priority = {'test':0,'validation':1,'train':2,'unspecified':3}
    df['_priority'] = df.original_split.map(priority)
    df = df.sort_values(['_priority','sample_id']).drop_duplicates('sample_id').drop(columns='_priority').reset_index(drop=True)
    df['provenance'] = df.sample_id.map(provenance)
    uf = UnionFind(len(df)); seen = {}; template = {}; edges=[]
    for i,row in df.iterrows():
        key = canonical(row.text)
        if key in seen: uf.union(i,seen[key])
        seen[key]=i
        if row.template_numbers:
            # Group patient/record-number variants from the known templated dataset.
            key = re.sub(r'\d+', '<NUMBER>',key)
            if key in template: uf.union(i,template[key])
            template[key]=i
    # Template-equivalent rows have identical group policy; compare representatives
    # plus all raw texts to avoid missing cross-source near matches.
    for a,b,score in near_pairs(df.text.tolist(),threshold):
        uf.union(a,b); edges.append({'left':df.iloc[a].sample_id,'right':df.iloc[b].sample_id,'jaccard':score})
    component_ids = {}
    for i, sid in enumerate(df.sample_id):
        root=uf.find(i);component_ids[root]=min(component_ids.get(root,sid),sid)
    df['group'] = [component_ids[uf.find(i)] for i in range(len(df))]
    return df, quarantine, {'exact_duplicates_removed':duplicates,'conflict_texts':len(conflicts),
        'exact_cross_split_texts':sum(len(s-{'unspecified'})>1 for s in original_splits.values()),
        'near_duplicate_edges':len(edges),'groups':df.group.nunique(), 'audit_only_nonconflicting_rows':audit_count,
        'mixed_label_groups':int((df.groupby('group').label.nunique()>1).sum()),
        'largest_groups':df.group.value_counts().head(10).to_dict()}, pd.DataFrame(edges)


def group_folds(df, n_splits=5, seed=42):
    counts = df.groupby('label').group.nunique()
    if set(counts.index) != {0,1} or counts.min() < n_splits:
        raise ValueError(f'Need >= {n_splits} distinct groups per class; got {counts.to_dict()}')
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(df.text,df.label,df.group))
    for a,b in folds:
        assert not set(df.iloc[a].group) & set(df.iloc[b].group)
        if df.iloc[a].label.nunique()!=2 or df.iloc[b].label.nunique()!=2:
            raise ValueError('Group constraints produced a single-class fold; reduce folds or obtain more groups.')
    return folds


def group_holdout(df, n_splits=5, seed=42):
    folds = group_folds(df,n_splits,seed)
    # Balance labels and size only, never inspect model performance.
    a,b = min(folds,key=lambda ab:abs(len(ab[1])/len(df)-1/n_splits)+abs(df.iloc[ab[1]].label.mean()-df.label.mean()))
    return df.iloc[a].reset_index(drop=True),df.iloc[b].reset_index(drop=True)


def assert_isolated(parts):
    for i,a in enumerate(parts):
        for b in parts[i+1:]:
            if set(a.group)&set(b.group) or set(a.sample_id)&set(b.sample_id):
                raise ValueError('Split leakage detected')


def partition(df, seed=42, validation_folds=5):
    df = df.copy(); dropped=[]
    # Never move official test examples to training. Drop linked development rows.
    test_groups=set(df.loc[df.original_split=='test','group'])
    mask=df.group.isin(test_groups)&df.original_split.ne('test')
    dropped.append(df[mask]); df=df[~mask]
    test=df[df.original_split=='test']; dev=df[df.original_split!='test']
    val_groups=set(dev.loc[dev.original_split=='validation','group'])
    mask=dev.group.isin(val_groups)&dev.original_split.ne('validation')
    dropped.append(dev[mask]);dev=dev[~mask]
    val=dev[dev.original_split=='validation'];train=dev[dev.original_split!='validation']
    if test.empty: train,test=group_holdout(train,5,seed)
    if val.empty: train,val=group_holdout(train,validation_folds,seed+1)
    parts=[]
    for name,p in [('train',train),('validation',val),('test',test)]:
        p=p.copy().reset_index(drop=True);p['split']=name
        if p.label.nunique()!=2: raise ValueError(f'{name} lacks both classes')
        parts.append(p)
    assert_isolated(parts)
    return pd.concat(parts,ignore_index=True),pd.concat(dropped,ignore_index=True)
