"""Summarize only persisted real results. Never fill missing experiments with estimates."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from promptguard.evaluation import comparison,metrics
from promptguard.utils import write_json

root=Path('runs/baseline');r=json.loads((root/'evaluation/tfidf/metrics.json').read_text())
prep=json.loads(Path('data/processed/preparation.json').read_text());inv=json.loads(Path('reports/inventory.json').read_text())
train=pd.read_json('data/processed/samples.jsonl',lines=True);train=train[train.split=='train'].reset_index(drop=True)
oof=pd.read_csv(root/'oof/oof_probabilities.csv').set_index('sample_id')
assert set(oof.index)==set(train.sample_id)
oof=oof.loc[train.sample_id]
manifest=json.loads((root/'oof/fold_manifest.json').read_text())
assert all(not set(f['fit_groups'])&set(f['holdout_groups']) for f in manifest)
fit_costs=[json.loads(p.read_text())['costs']['wall_seconds'] for p in (root/'oof').glob('fold_*/*/model/complete.json')]
oof_info={'rows':len(oof),'columns':oof.columns.tolist(),'missing':int(oof.isna().sum().sum()),
          'folds':len(manifest),'total_nested_training_seconds':sum(fit_costs),
          'fixed_0_5_threshold_metrics':metrics(train.label,oof.tfidf.to_numpy(),.5),
          'scope':'TF-IDF only. Not a complete four-model OOF matrix. No meta-classifier has been trained on real data.'}
write_json(root/'oof/real_oof_summary.json',oof_info)
comparison([r],root/'evaluation')
unchanged=all(hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256'] for f in inv['files'])
write_json('reports/original_integrity.json',{'all_inventory_files_match_sha256':unchanged,'files_checked':len(inv['files'])})
t=r['test'];v=r['validation'];cost=r['costs']
status={'dataset_audit':'completed','tfidf':'trained and evaluated on real local data','tfidf_oof':'completed; 5 nested group-aware folds',
        'transformers':'not trained: requested pretrained weights unavailable locally; package-network DNS unavailable',
        'ensembles_and_full_ablations':'implementation integration-tested; no real training or performance results',
        'family_held_out':'not supported by currently admitted positive labels','tests':'19 passed; 3 upstream SciPy deprecation warnings',
        'original_datasets_unchanged':unchanged}
write_json('reports/status.json',status)
lines=['# Research results and completion status','',
'Only measurements from the real local datasets are reported below. Random tiny Transformer models were used exclusively in unit/integration tests; their scores are not research results. No external datasets were downloaded.',
'', '## Final corrected dataset','',
'Eleven data files across six collections were inspected. CSV/Parquet mirrors were audited rather than added as independent examples. The benchmark versions share 4,981 texts, disagree on eight shared labels, and each has 19 distinct texts. The Parquet revision is canonical; eight disputed texts are quarantined. Original files remain unchanged (SHA-256 verified).',
'', '| Split | Benign | Injection | Total | Groups |','|---|---:|---:|---:|---:|']
for name in ['train','validation','test']:
 p=prep['partitions'][name];lines.append(f"| {name} | {p['labels']['0']} | {p['labels']['1']} | {p['rows']} | {p['groups']} |")
lines += ['',f"Preparation removed {prep['exact_duplicates_removed']:,} exact duplicate rows from admitted canonical sources and quarantined {prep['conflict_texts']} conflicting texts across benchmark versions. {prep['near_duplicate_edges']} verified lexical near-duplicate links were grouped. Known patient-number templates were also grouped. {prep['cross_split_rows_quarantined']} development rows overlapped final test groups. Sixty thousand C.A.M.S rows have unverified/ambiguous labels; safety, the small numeric-label collection, and the attack-task dataset remain excluded entirely.",
'', 'The safety source contains 800 shared exact texts between its supplied train/test files. It must not be added without global contamination removal. The task dataset IDs repeat across task types; future derived-label admission must group related target/injection variants, not treat each column as independent.',
'', '## Actual TF-IDF results','',
'Hyperparameters were selected with three group-aware folds inside training. The compact planned search compared word unigrams/bigrams and character 3–5 grams, vocabulary caps, min_df, regularization and class weighting. The selected final pipeline uses word unigrams/bigrams, 20,000 maximum features, min_df=1, C=1, and no class weighting. Case is preserved.',
'', '| Metric | Validation | Test |','|---|---:|---:|']
for key in ['accuracy','precision','recall','f1','macro_f1','roc_auc','average_precision','pr_auc_trapezoid','false_positive_rate','false_negative_rate','brier_score','ece_10_equal_width']:
 lines.append(f'| {key} | {v[key]:.6f} | {t[key]:.6f} |')
lines += ['',f"Threshold = **{t['threshold']:.9f}**, selected by validation macro F1. No test threshold, weight, or hyperparameter tuning was performed.",
'', 'Confusion matrix (rows true benign/injection, columns predicted benign/injection):', '',
'```text',f"{t['confusion_matrix'][0]}",f"{t['confusion_matrix'][1]}",'```','',
'### Interpretation and error analysis','',
'**The baseline fails to generalize reliably.** Perfect validation performance coexists with a very high external false-positive rate. Development class and source are confounded: all admitted benign training samples are templated medical queries and all positive training samples are from a separate multilingual injection collection. This is not evidence of a successful security detector.',
'Coefficient inspection supports this concern: `patient`, `patients`, `for patients`, and `Compare` strongly favor benign, while common words such as `your`, `de`, and `my` favor injection. The complete top-feature audit is in `reports/tfidf_feature_audit.csv`.',
'False positives include a benign request to convince a reader to attend a meeting, and an educational TOEFL article translation request. These are qualitative examples, not newly assigned attack categories. One missed positive asks about previous instructions; another embeds an instruction override before a healthcare question. Labels are source annotations, not infallible security judgments. All errors and probabilities are exported for review.',
'No dedicated benign-security-quotation category is provided in the admitted data, so a numerical claim for that category would be unsupported. Current calibration metrics also reflect domain shift; no post-hoc calibrator was fitted.',
'', '### Computational cost','',
f"Final pipeline CV search plus refit: {cost['wall_seconds']:.3f} seconds. Artifact size: {cost['model_bytes']:,} bytes. Warm single-request p50/p95: {cost['single_ms_p50']:.3f}/{cost['single_ms_p95']:.3f} ms. Batch-16 mean: {cost['batch_ms_per_sample_mean']:.4f} ms per sample. Latency includes vectorization and prediction, excludes loading, and was measured on 128 validation texts; it is not a long-payload throughput guarantee.",
f"Peak process RSS: {cost['process_peak_rss_mib']:.1f} MiB, cumulative across imports/data preparation in that process, not isolated model RAM. Hardware: AMD Ryzen 5 5500U, 12 logical CPUs, approximately 18.9 GiB RAM, no CUDA device. CPU thread limits were 4 for the real baseline runs.",
'', '### Real OOF experiment','',
f"Five outer folds produced {len(oof):,} TF-IDF probabilities with exactly one prediction per training sample and no missing entries. Each fold runs its own three-fold hyperparameter search before predicting its held-out groups. Total nested fitting time: {sum(fit_costs):.3f} seconds. Fold sample/group identities and fitted models are retained. This is one column, **not** the requested complete four-column stacking matrix.",
'', '## Remaining experiments and implementation coverage','',
'| Experiment | Actual result status |','|---|---|',
'| TF-IDF logistic regression | Trained and evaluated above |',
'| BERT | Not trained: pretrained assets unavailable |',
'| RoBERTa | Not trained: pretrained assets unavailable |',
'| DistilBERT | Not trained: pretrained assets unavailable |',
'| RoBERTa + TF-IDF weighted ensemble | Implemented; not run with real trained models |',
'| Transformer-only weighted ensemble | Implemented; not run with real trained models |',
'| Four-model weighted ensemble | Implemented; not run with real trained models |',
'| Four-model stacking ensemble | Implemented; not run with real trained models |',
'', 'The complete OOF and disjoint-holdout workflows were integration-tested with tiny local model fixtures, including all eight result artifacts, inference and checkpoint consistency. The code provides fresh fold initialization, inner-only selection, final refits, probability ordering, weight constraints, validation thresholds, plotting, disagreement analysis and paired group-bootstrap ablation intervals. No ensemble effectiveness conclusion can be drawn yet.',
'Five-fold real training entails 15 outer Transformer fits plus three final fits; inner epoch selection and fresh refits make up to 36 training passes. Timing requires real-model measurement. Full GPU configuration and a lower-cost holdout alternative are provided. Missing weights fail preflight explicitly.',
'Positive attack-family metadata is unavailable under the verified admission policy. The family-held-out TF-IDF runner is implemented but no such evaluation was performed. No results are claimed for role spoofing, indirect attacks, encoding, multilingual categories, benign security quotations, long-context or subtle attacks.',
'', '## Verification and next prerequisites','',
'- 19 automated tests passed, including full OOF and holdout integration. Three warnings come from an upstream SciPy solver deprecation.',
'- A separate smoke subset sampled real rows within existing isolated partitions; its TF-IDF training and inference also ran. Smoke scores are not promoted to research conclusions.',
'- Original dataset file hashes match the inventory.',
'- Pretrained model/tokenizer files are needed for the exact requested models. Network-enabled model download or local checkpoint paths can unblock training; no substitute dataset/model is used.',
'- Verified numeric-label definitions and C.A.M.S attack-class definitions are needed before those sources can enter binary training. Broader benign training coverage is essential to address the observed source shortcut.',
'', '## Artifacts','',
'- `reports/dataset_analysis.md`, `reports/inventory.json`: complete audit and mapping policy.',
'- `data/processed/`: frozen split manifest, raw-text samples, provenance and quarantines.',
'- `runs/baseline/models/tfidf/`: actual model, CV results and cache identity.',
'- `runs/baseline/evaluation/tfidf/`: metrics, per-class scores, errors, predictions, slices and standalone plots.',
'- `runs/baseline/oof/`: five real fold models, split identities and OOF probabilities.',
'- `runs/full/preflight.json`: documented missing pretrained models.',
'- `reports/tests.log`, `reports/environment.freeze.txt`, `reports/original_integrity.json`: verification records.',
'', 'The earlier exploratory run in `runs/baseline_initial_unresolved_labels` is explicitly superseded. The final run uses the corrected benchmark conflict policy and stable group IDs. Hyperparameter search space and threshold objective were not changed in response to test metrics. Since the benchmark has now been inspected during development, any future performance-driven redesign should use a newly reserved evaluation corpus with verified labels.']
Path('reports/research_results.md').write_text('\n'.join(lines)+'\n')
print(status)
