#!/bin/sh
# Run from any directory; keep generated results owned by the invoking user.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
mkdir -p docker-output/runs docker-output/reports docker-output/processed docker-output/cache
if [ "${PROMPTGUARD_SKIP_BUILD:-0}" = 1 ]; then
    exec docker compose run --rm promptguard "$@"
fi
exec docker compose run --build --rm promptguard "$@"
