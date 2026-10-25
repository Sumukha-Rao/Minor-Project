import json
from pathlib import Path
from .data import inventory, load_verified
from .splitting import deduplicate_and_group, partition
from .utils import hardware, write_json, fingerprint


def prepare(config):
    out=Path(config['prepared_dir']);out.mkdir(parents=True,exist_ok=True)
    report_dir=Path('reports');report_dir.mkdir(exist_ok=True)
    inv=inventory(config['dataset_root'],report_dir)
    raw,quarantine=load_verified(config)
    clean,conflicts,audit,edges=deduplicate_and_group(raw,config['near_duplicate_jaccard'])
    split,dropped=partition(clean,config['seed'],config['validation_folds'])
    split.to_json(out/'samples.jsonl',orient='records',lines=True,force_ascii=False)
    split[['sample_id','group','split','source','label','category']].to_csv(out/'split_manifest.csv',index=False)
    quarantine.to_csv(out/'unverified_labels.csv',index=False)
    conflicts.to_json(out/'label_conflicts.jsonl',orient='records',lines=True)
    dropped.to_json(out/'cross_split_quarantine.jsonl',orient='records',lines=True)
    edges.to_csv(out/'near_duplicate_edges.csv',index=False)
    summary={'admitted_rows_before_dedup':int((~raw.audit_only).sum()),'unverified_rows_in_selected_sources':len(quarantine),
        'cross_split_rows_quarantined':len(dropped),**audit,'fingerprint':fingerprint(split),
        'partitions':{name:{'rows':len(g),'labels':g.label.value_counts().to_dict(),'groups':g.group.nunique(),
                          'source_by_label':g.groupby(['source','label']).size().to_dict()}
                      for name,g in split.groupby('split')}}
    # Tuple keys cannot be JSON keys.
    for part in summary['partitions'].values(): part['source_by_label']={str(k):v for k,v in part['source_by_label'].items()}
    write_json(out/'preparation.json',summary);write_json(out/'config.json',config)
    write_json(report_dir/'hardware.json',hardware())
    lines=['# Dataset analysis (before training)','',
        'Original files are read only. See inventory.json for schemas, missing values, lengths, label examples, duplicate counts, and cross-file overlaps.',
        '', '| File | Rows | Labels |', '|---|---:|---|']
    for entry in inv['files']:
        lines.append(f"| {entry['path']} | {entry.get('rows','unreadable')} | {entry.get('labels',entry.get('error','unlabeled'))} |")
    lines += ['', '## Verified admission policy','',
        '- Benchmark: benign → 0; jailbreak → 1, per local README. Parquet is the canonical test revision. CSV is audit-only: conflicting labels on shared texts are quarantined; CSV-only rows are not added. Jailbreak detection is a related but broader construct.',
        '- prompt_injections: all rows → 1, per local README.',
        '- C.A.M.S: literal legitimate → 0; ambiguous/mextra/minja quarantined. Patient number variants share groups.',
        '- Safety: 0/1/2 have no documented meanings in local README; entire source excluded.',
        '- hugging face: 0/1 lack documented meanings and examples include ordinary instructions labeled 1; excluded.',
        '- Attack dataset: target_label and inject_label are task outcomes, not detector labels. All attack variants and task metadata inventoried; excluded pending an explicit derived-label protocol.',
        '', '## Leakage policy','',
        'Exact raw-text label conflicts are quarantined. Duplicate rows retain provenance. NFKC/case/whitespace canonicalization is used only for grouping. Near duplicates are connected components of verified character 5-shingle Jaccard ≥ '+str(config['near_duplicate_jaccard'])+'. The prefix join is exhaustive under that lexical definition; semantic paraphrases may remain. Known numeric templates are grouped. Training text is not normalized.',
        'Official test rows stay test. Development rows connected to test are quarantined. Remaining data is split by stratified groups. TF-IDF vocabulary and hyperparameters use training folds only. Thresholds use validation only.',
        '', '## Critical limitations','',
        '**Development class and source are confounded:** currently admitted benign training data comes from C.A.M.S and malicious training data from prompt_injections. High validation metrics may reflect source identification. The external benchmark tests domain shift, not comprehensive prompt-injection security. Additional documented binary data is required for credible conclusions.',
        'No language, encoded-payload, long-context, educational-benign, or subtle-attack labels are invented. Available C.A.M.S subcategories are preserved but currently contain only admitted benign examples. Family-held-out attack evaluation is unavailable with this admission policy.',
        '', '## Preparation results','', '```json', json.dumps(summary,indent=2,default=str),'```']
    (report_dir/'dataset_analysis.md').write_text('\n'.join(lines))
    return summary
