#!/bin/bash
set -e

export PATH="$HOME/.cargo/bin:$PATH"

# Load .env from project root
set -a
source "$(dirname "$0")/../.env"
set +a

# Override for host mode (infra running in Docker, Django on host)
export DJANGO_DEBUG=1
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5434
export STORAGE_HOST=localhost:9000
export REDIS_HOST=localhost

cd "$(dirname "$0")/../aquillm"
uv run python manage.py runserver 0.0.0.0:${PORT:-8080}
