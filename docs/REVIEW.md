# Project review and delivery assessment

Reviewed source revision: cbf6720783ca02240ed14c3c7c9ae54aeddc5d86 (main), plus the Docker/documentation changes in this working tree. Review date: 26 October 2026, Asia/Kolkata. The review covers the Python implementation, configuration, tests, data policy, archived reports and container packaging. It is not an independent relabeling of the datasets or a security certification.

## Executive assessment

PromptGuard is a local-data research system for binary prompt-injection detection. It combines a lexical TF-IDF/logistic-regression baseline with BERT, RoBERTa and DistilBERT classifiers, then supports weighted averaging and logistic stacking. Its strongest engineering feature is explicit separation of training, validation and test groups: preprocessing, nested selection and ensemble fitting aim to prevent near-duplicate contamination and in-sample stacking.

The project is a command-line research application. It has no REST API, dashboard, SQLite database, Mobile IP simulator or network handover functionality. Those would be separate projects. Its normal outputs are model artifacts, JSON/CSV measurements and self-contained Plotly HTML charts.

The key research limitation is already visible in the archived evidence: a TF-IDF model achieved perfect validation scores but flagged 93.79% of benign external test examples as injection. This is consistent with source/class confounding. Adding Transformer models does not establish that the underlying data problem is solved. The current system is suitable for demonstrating methodology and failure analysis, not as a validated standalone production security boundary.

## Findings in priority order

| Priority | Finding and evidence | Consequence / recommendation |
|---|---|---|
| High | `reports/research_results.md` reports test accuracy 0.436498, macro F1 0.351530 and false-positive rate 0.937896, despite validation macro F1 of 1.0. Verified training positives and negatives originate from different sources. | Broad benign coverage and independently verified attack labels are needed. Reserve a new evaluation corpus before performance-driven redesign; do not tune against the already inspected benchmark. |
| High | Trained bundles and `data/processed` are ignored and absent from a fresh clone, although historical README passages describe them as available. | Docker smoke now trains before inference; the new guide states exactly where generated artifacts live. Existing historical documentation is retained as historical evidence. |
| Medium | `oof.code_fingerprint` and `experiment.fit_base` guard code/config/data identity but omit dependency versions and local checkpoint content hashes. | Replacing a checkpoint at the same path or changing dependencies can reuse a cache with an insufficient identity. Preserve image/freeze/checkpoint hashes and use a new run directory. Stronger identity checks are recommended follow-up work. |
| Medium | `config.load_config` validates stacking mode and fold count but most numerical ranges, required source fields and paths are checked only downstream. | Malformed configuration can fail late. Add comprehensive schema validation before costly work in a separate code change. |
| Medium | `experiment.preflight` returns availability records; the CLI can exit zero even when every model is unavailable. | Automation must inspect each `available` field. The full experiment correctly raises on missing models. A strict preflight exit mode would improve orchestration. |
| Medium | `data.inventory` records per-file exceptions instead of failing the entire command. | A successful inspection command does not imply every dataset was readable. Inspect each file's `error` entry. Preparation still fails for unavailable configured sources. |
| Low | `data.read_table` advertises `.xls`, but the requirements contain openpyxl rather than xlrd. | Modern `.xlsx` is supported; legacy `.xls` may fail. This does not affect the repository's CSV/Parquet workflow. Add/test xlrd only if legacy Excel becomes required. |
| Low | Project requirements use version ranges and the baseline environment freeze contains machine-specific CUDA packages. | Docker adds a tested CPU set of direct constraints. Full bitwise reproducibility still needs a transitive hash lock and immutable base/image digests. |

The review documents these issues; it does not silently rewrite model selection, dataset admission or experimental logic. Containerization changes execution packaging, not the scientific conclusions.

## Architecture and component responsibilities

| Component | Responsibility and interface |
|---|---|
| `cli.py`, `__main__.py` | Parse commands and dispatch inspect, prepare, train-base, oof, run, preflight, predict and family-holdout. |
| `config.py` | Read JSON/YAML settings and basic mode/fold validation. Paths are relative to the working directory. |
| `data.py` | Local CSV/JSON/JSONL/Parquet/Excel readers, file inventory, explicit label admission, unknown-label quarantine. Dataset text is untrusted input, never operational instructions. |
| `splitting.py` | Hash samples, preserve raw text, group exact/lexical/template duplicates, protect official test groups and construct grouped folds. |
| `prepare.py` | Coordinate inventory, admission, grouping, partitioning and export immutable split/data manifests. |
| `tfidf.py` | Fold-local vocabulary and IDF fitting with logistic regression and grouped hyperparameter search. |
| `transformer.py` | Load exact pretrained assets, tokenize, select epochs inside training, refit, checkpoint/resume and predict. Remote custom model code is disabled. |
| `oof.py` | Generate exactly one held-out probability per training row/model; persist fold identities and validate caches. |
| `ensemble.py` | Enforce probability column order; fit logistic stacking and nonnegative simplex weights with explicit provenance. |
| `evaluation.py` | Select thresholds on validation macro F1; compute test metrics, error exports, calibration and standalone plots. |
| `robustness.py` | Metadata-based slices, disagreements, grouped bootstrap and family-held-out baseline analysis. |
| `experiment.py` | Orchestrate individual fits, OOF or holdout ensembles, availability checks and bundle creation. |
| `inference.py` | Load trusted saved bundles; predict single strings/batches; reject invalid input and model-path escape. |
| `utils.py` | Seeds, hashing, atomic JSON writes, hardware metadata and elapsed-time/resource measurements. |
| `tests/` | Data integrity, model/checkpoint behavior, stacking guards and full tiny-model orchestration. |
| `scripts/` | Historical report generation, local wheel recovery and smoke subset construction. Wheel recovery is not needed in Docker. |
| `Dockerfile`, `compose.yaml`, `docker/` | Reproducible CPU packaging, safe bind layout and convenience workflows. |

The data flow is: local files → audited label admission → duplicate groups → fixed train/validation/test → fold-local base fits → held-out probability columns → ensemble fitting → validation threshold → frozen-test reports → trusted inference bundle. Only training rows train model parameters; only held-out training predictions train stackers; validation chooses the operating threshold; test reports outcomes.

## Algorithm explanation

TF-IDF transforms token or character counts into sparse numerical features. Term frequency is sublinear and IDF reduces the weight of common terms. Logistic regression maps the weighted feature sum through a sigmoid to estimate the positive-class probability. The compact grid tests four word/character and regularization settings. Crucially, vectorization is inside the scikit-learn pipeline, so a held-out fold does not contribute its vocabulary or IDF.

Transformer classifiers tokenize raw text and use contextual representations from pretrained BERT, RoBERTa or DistilBERT with a two-class head. BERT's uncased tokenizer inherently lowercases; the pipeline otherwise preserves original text. Dynamic padding reduces wasted compute, while truncation limits context. AdamW, learning-rate warmup, gradient accumulation, clipping, optional CUDA mixed precision and class weighting are implemented. Epoch selection uses an inner group-disjoint split, followed by a fresh pretrained initialization and refit on all data supplied to that fit. Real-model accuracy remains unmeasured until those runs are executed.

OOF means out-of-fold: each training example receives a probability from a model that did not train on that example's group. Four columns have the fixed order BERT, RoBERTa, DistilBERT, TF-IDF. Logistic stacking learns a second classifier over these columns. Weighted averaging minimizes log loss subject to nonnegative weights summing to one. Neither weights nor the meta-classifier may be trained on the final test partition. The lower-cost holdout mode reserves separate training rows for meta-fitting and deliberately avoids later refitting base models on those rows.

Thresholds convert probabilities into labels. This implementation searches distinct validation probabilities and an all-negative candidate, maximizing macro F1 with a tie preference near 0.5. A high recall can coexist with a very poor false-positive rate; therefore the report includes per-class metrics, ROC/PR information, calibration, confusion matrices and errors rather than a single accuracy number.

## Security, privacy and operational limits

Pickle/joblib and training resume files are trusted local artifacts only; loading an untrusted model can execute code. Bundle path checks reduce path traversal but do not make untrusted deserialization safe. Raw prompts are preserved in prediction/error exports and may contain sensitive data. Keep output directories private and review them before sharing. The Docker context excludes raw full datasets and outputs, but includes the real smoke subset; licensing still applies to image distribution.

The detector sees text, not application trust boundaries. A malicious instruction and a benign quotation can be textually similar. Domain shift, truncation, obfuscation and missing language/attack-family annotations constrain claims. Probability scores are not guaranteed calibrated risks. No production throughput, adversarial robustness, GPU behavior or real ensemble superiority is claimed by this packaging work.

## Reproduction and demonstration plan

1. Build and run `./docker/run.sh smoke`; explain that the subset is a wiring check.
2. Run `./docker/run.sh test`; distinguish random tiny-model integration fixtures from pretrained research experiments.
3. Run `./docker/run.sh baseline`; inspect fresh inventory errors and split manifests before interpreting metrics.
4. Open the fresh TF-IDF evaluation HTML and explain the validation/test performance gap.
5. Predict one benign and one suspicious example. Do not interpret two examples as accuracy evidence.
6. Run preflight and inspect per-model availability. Real ensemble comparison requires actual pretrained assets and training.
7. Preserve image ID, dependency freeze, configurations, data hashes and generated run directories with any submitted results.

The companion Docker guide supplies exact commands and mount paths. The consolidated DOCX also includes the existing detailed dataset processing record, the project guide, original dataset source revisions, archived results and new execution evidence. Historical machine paths and measurements in those appendices are clearly attributed to the previous run; use the Docker guide for this delivered workflow.
