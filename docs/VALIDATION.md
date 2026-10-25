# Current container validation

These checks were executed in the cloud Docker engine against this delivery, on 26 October 2026 (Asia/Kolkata). They are distinct from the historical report appended later.

## Build and runtime

The Python 3.12 / PyTorch 2.6.0 CPU image built successfully, including `pip check`. Image ID: `sha256:01fc4794498d96750a0d987c2e5de81c42bd447f22290cddb505bc872a805d63`. Reported image size is 1,663,988,817 bytes (about 1.55 GiB); Docker storage overhead is additional. A clean plain-internet build was not tested: this cloud required a writable Docker client config, host-network build, explicit mapping for its proxy hostname, HTTP/HTTPS build proxy arguments, and the existing trusted CA bundle supplied through a BuildKit secret. TLS checks remained enabled. No proxy credentials or private values were persisted in project files.

The first runtime check exposed source directory permissions inherited from this cloud checkout. The Dockerfile now makes its copied application tree readable/traversable before switching to UID 1000. Final validation used the non-root Compose service, read-only root filesystem, dropped capabilities and the documented output binds. No network ports or privileged mode were needed.

| Check | Actual outcome |
|---|---|
| Docker image build and pip dependency check | Passed |
| Compose configuration validation | Passed |
| Existing automated suite in container | 19 passed in 23.32 seconds; none skipped |
| Real-data smoke training, evaluation and prediction | Passed; output files persisted on host |
| Second smoke invocation through helper | Passed; matching model cache reused |
| Full local-data inventory | 11 files read, zero recorded reader errors |
| Full preparation and TF-IDF training/evaluation | Passed |
| Nested TF-IDF OOF | Five folds, 9,249 unique rows, finite valid probabilities, disjoint fit/holdout samples and groups |
| Original dataset integrity | All 11 inventoried file SHA-256 values matched after execution |
| Real pretrained model preflight | BERT, RoBERTa and DistilBERT unavailable in the empty offline cache |
| Real Transformer/ensemble training | Not run; no research performance claim |
| GPU support | Not tested; delivered image is CPU-only |
| DOCX verification | Generated with python-docx and reopened structurally; Word/LibreOffice visual rendering not performed |

`preflight` exited zero while returning `available: false`, consistent with the documented CLI behavior. Defaults request local assets only, so this result establishes missing local checkpoints, not a tested denial of Hugging Face download access. Tiny random Transformer fixtures passed in integration tests; their predictions are not research evidence.

## Fresh full-data results

| Metric | Validation | External test |
|---|---:|---:|
| Samples | 2,091 | 4,992 |
| Accuracy | 1.000000 | 0.436498 |
| Macro F1 | 1.000000 | 0.351530 |
| False-positive rate | 0.000000 | 0.937896 |

Test confusion matrix, rows true benign/injection and columns predicted benign/injection:

```text
             Predicted benign    Predicted injection
True benign          186                  2809
True injection         4                  1993
```

These classification results reproduce the archived baseline at the reported precision. High recall does not offset 2,809 false alarms among 2,995 benign texts. This supports the review's warning about source/class confounding. New hardware latency should be read from the fresh metrics JSON; historical timing is not carried over as a current benchmark. The small smoke model returned probability 0.632705826589432 for the supplied suspicious example; that single prediction demonstrates inference, not effectiveness.

## Evidence and reproducibility

Fresh logs and dependency versions are saved under `reports/container-validation/`: `build.log`, `tests.log`, `smoke.log`, `repeat.log`, `baseline.log`, `preflight.log`, `environment.freeze.txt`, and `summary.json`. Full models, prepared data, prediction CSVs and interactive evaluation HTML remain in ignored `docker-output/` directories. These large generated artifacts are not included in Git; the Docker commands regenerate them. Repeat runs with matching fingerprints reuse completed fits and are not independent experimental repetitions.

Current validation used `docker compose run --rm promptguard test`, `smoke`, and `baseline` after the managed-proxy image build. The documented helper was exercised using `PROMPTGUARD_SKIP_BUILD=1` for the repeat smoke and preflight checks. A plain-internet `./docker/run.sh smoke` path uses the same image/workflow but its default build networking was not available in this cloud.

The deliverable changes Docker packaging, the README quick start, ignore rules, documentation and the DOCX generation script. Existing model logic, tests, dataset files and baseline configuration remain unchanged. Remaining scientific work includes broader verified benign data, confirmed labels for excluded sources, real pretrained training, a fresh held-out corpus and supported robustness annotations.
