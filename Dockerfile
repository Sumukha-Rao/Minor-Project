FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    HOME=/tmp \
    HF_HOME=/app/.cache/huggingface
WORKDIR /app

# CPU wheels avoid installing unused CUDA libraries. Keep TLS verification enabled.
COPY requirements.txt pyproject.toml ./
COPY docker/constraints.txt /tmp/constraints.txt
RUN --mount=type=secret,id=build_ca \
    if [ -s /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt -c /tmp/constraints.txt
COPY src/ src/
RUN --mount=type=secret,id=build_ca \
    if [ -s /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    pip install --no-cache-dir --no-deps . && pip check
COPY configs/ configs/
COPY tests/ tests/
COPY data/smoke/ data/smoke/
COPY docker/entrypoint.py docker/entrypoint.py
RUN mkdir -p runs reports data/processed .cache/huggingface \
    && chmod -R a+rX /app \
    && chown -R 1000:1000 runs reports data/processed .cache
USER 1000:1000
ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD ["smoke"]
