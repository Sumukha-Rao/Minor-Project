# Docker operation guide

This repository is a Python CLI research pipeline, not a web application. No web server, port, database service, or privileged container is required. The default image runs on CPU; real Transformer training still requires pretrained checkpoints and considerable compute.

## Prerequisites and quick start

Install Docker Engine with Compose v2 on Linux, or Docker Desktop on macOS/Windows. Use a POSIX shell (WSL on Windows). Run from a complete Git clone, including the tracked datasets. Allow several GB of free disk space for the CPU image and additional space for experiments. The first build needs access to Docker Hub, PyPI, and the official PyTorch CPU wheel index.

```bash
./docker/run.sh smoke
./docker/run.sh test
./docker/run.sh baseline
```

The helper builds the image, creates output directories, runs using your host UID/GID, and removes the finished container. `smoke` trains and evaluates TF-IDF on the included small real-data subset, then predicts one example. Its scores check execution only. `test` runs the existing automated suite, including tiny random Transformer fixtures. `baseline` audits and prepares the original datasets, trains TF-IDF, and generates nested out-of-fold predictions. It can take longer than smoke.

If you already built/imported the image (for example using the proxy-aware build below), run `PROMPTGUARD_SKIP_BUILD=1 ./docker/run.sh smoke`. This skips only the image build; it still runs the workflow and preserves failures. Rebuild after source/dependency changes.

On PowerShell, create the bind directories before invoking Compose:

```powershell
New-Item -ItemType Directory -Force docker-output/runs,docker-output/reports,docker-output/processed,docker-output/cache
 docker compose run --build --rm promptguard smoke
```

On Linux, prefer the helper to avoid UID mismatches. Do not use sudo to run the helper merely to fix ownership. Docker access must already be configured for your account. Docker access itself grants substantial control over the host.

## Commands and outputs

```bash
./docker/run.sh --help
./docker/run.sh --config configs/baseline.json prepare
./docker/run.sh --config configs/baseline.json train-base tfidf
./docker/run.sh --config configs/baseline.json oof --models tfidf
./docker/run.sh predict --run-dir runs/baseline --model tfidf --text 'Summarize the meeting agenda.'
./docker/run.sh preflight
```

The global `--config` option must precede the project subcommand. Docker conveniences (`smoke`, `baseline`, `test`) accept no additional arguments; use the native CLI for customization. A command failure propagates a nonzero exit code. Preflight is an upstream exception: its process may succeed while model availability is false; inspect its JSON.

| Host path | Container path | Purpose |
|---|---|---|
| prompt injection dataset/ | /app/prompt injection dataset/ | Original datasets, read only |
| configs/ | /app/configs/ | Configuration, read only |
| docker-output/processed/ | /app/data/processed/ | Prepared full dataset and split manifests |
| docker-output/runs/ | /app/runs/ | Trained models, predictions, evaluation HTML |
| docker-output/reports/ | /app/reports/ | Fresh inventory, data audit and hardware |
| docker-output/cache/ | /app/.cache/huggingface/ | Pretrained model cache |

The small tracked `data/smoke` is copied into the image. No full datasets, Git metadata, host virtualenvs, private files, or existing trained models are copied into the image. The smoke subset is real dataset content: check dataset licensing before distributing an image. Source and tests are baked in; changing them requires a rebuild (the helper requests one, reusing cached layers).

Open `docker-output/runs/baseline/evaluation/tfidf/evaluation.html` in a browser after baseline completion. No HTTP service is needed. Copy the entire run directory to preserve relative inference paths. Keep config, prepared data and image/dependency records with results. Do not commit generated raw-text predictions or model outputs unintentionally.

## Real Transformer and ensemble runs

Copy `configs/default.json` to `configs/docker-full.json`. Keep `prepared_dir` as `data/processed`; set a new `run_dir`, for example `runs/docker-full-01`. To download the exact configured pretrained assets, set `transformer.local_files_only` to `false`. The models are BERT, RoBERTa and DistilBERT, as named in the original config. This downloads model assets, not datasets. An offline alternative is to populate the mounted cache with complete matching model/tokenizer files or use complete local checkpoint directories under that mount and update model paths accordingly.

```bash
./docker/run.sh --config configs/docker-full.json preflight
# Continue only if all three available fields are true.
./docker/run.sh --config configs/docker-full.json run
./docker/run.sh predict --run-dir runs/docker-full-01 --model stacking --text 'Text to classify'
```

Prepare the full dataset first using `baseline` or the explicit `prepare` command. Required model destinations can include huggingface.co and its asset/CDN hosts. If behind a controlled network, configure access according to actual failed destinations. Never disable certificate verification. The supplied image is CPU-only; a GPU JSON preset does not add CUDA to it. A CUDA image, compatible driver and NVIDIA Container Toolkit would need a separately validated extension. No GPU support is claimed here.

A full five-fold ensemble can require up to 36 Transformer training passes, counting inner epoch selection and fresh refits. Do not treat tiny-fixture integration tests as a substitute for these experiments. Repeated completed runs reuse caches. Change `run_dir` for any changed code, data or settings; do not remove cache checks to force incompatible reuse. Back up outputs before changing configurations.

## Container design and reproducibility

The image uses Python 3.12 on Debian Bookworm, installs the official CPU PyTorch wheel first, then the project requirements with explicit direct-version constraints. `pip check` validates package consistency during build. Direct constraints are not a complete transitive lockfile; the base tag and transitive packages may change. For archival reproduction, keep the built image digest and a freeze record:

```bash
docker image inspect promptguard-research:local --format '{{.Id}}'
docker compose run --rm --entrypoint python promptguard -m pip freeze > docker-output/reports/container.freeze.txt
```

Compose drops capabilities, sets no-new-privileges, uses a read-only root filesystem and a bounded temporary filesystem, and exposes only required writable output/cache mounts. The image defaults to UID 1000; the helper maps your current UID/GID. Running the helper as root runs the container as root too. There is no restart policy because jobs should terminate and report their status.

## Troubleshooting

| Symptom | Action |
|---|---|
| Docker permission denied or daemon unavailable | Start Docker and configure access for your account. |
| Bind output permission denied | Use the helper; verify host directory ownership. Do not make the project world writable. |
| Container DNS cannot reach package servers | Fix Docker DNS or supply your supported HTTP/HTTPS proxy settings at build time. |
| Corporate/proxy CA not trusted | Supply a trusted PEM bundle as BuildKit secret `build_ca`; keep TLS checks enabled. |
| Read-only Docker client home | Set `DOCKER_CONFIG` to a writable directory, e.g. `/tmp/promptguard-docker-config`. |
| Model/checkpoint missing | Run training first, or populate the expected pretrained cache; Git does not contain trained bundles. |
| Stale model cache | Preserve the old output; use a distinct `run_dir` after code/data/config changes. |
| Baseline flags many ordinary prompts | Known dataset/generalization problem, not a Docker installation failure. |
| Out of memory during real Transformer work | Reduce batch/token settings in a new configuration; use more capable hardware. |

For a managed proxy with a custom CA, Docker supports:

```bash
docker build --build-arg HTTPS_PROXY --build-arg HTTP_PROXY --build-arg NO_PROXY \
  --secret id=build_ca,src=/path/to/trusted-ca-bundle.pem \
  -t promptguard-research:local .
```

The proxy environment variables are supplied by your machine; do not write their values into the Dockerfile or commit credentials. The optional CA secret is used only during package installation and is not retained in the image. Runtime model downloads behind such a proxy need equivalent runtime proxy and CA configuration. Do not put secrets into an image or command history. A host-network build or explicit proxy-host mapping may be necessary in managed cloud environments; these are environment-specific diagnostics, not portable defaults.

Stopping a job: interrupt the foreground command. Its container is removed; completed output files remain. `docker compose down` removes Compose resources but does not remove bind-mounted results. Delete or archive only selected output directories after confirming they are no longer needed.

## Rebuild the consolidated report

The checked-in DOCX is generated from the Markdown review, Docker guide, validation record, project guide, dataset-processing explanation, archived results and source register. To regenerate it after updating those sources:

```bash
python -m pip install -r docs/requirements-report.txt
python scripts/build_project_report.py
```

The generator writes `docs/PROJECT_REPORT.docx`, with styled headings, tables, code blocks, page numbers and a contents overview. It validates that the generated OOXML can be reopened. It does not require Microsoft Word or a running container.
