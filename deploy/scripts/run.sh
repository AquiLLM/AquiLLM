#!/bin/bash
set -e

# Compiled assets live outside /app so a development bind mount cannot hide them.
test -s "${AQUILLM_BUILT_STATIC_DIR:-/opt/aquillm-static}/js/dist/main.js"

cd /app/aquillm

/opt/venv/bin/python ./manage.py migrate --noinput
/opt/venv/bin/python ./manage.py collectstatic --noinput
if [ "${RUN_CELERY_IN_WEB:-1}" = "1" ]; then
  /opt/venv/bin/celery -A aquillm worker --loglevel=info &
fi
exec /opt/venv/bin/python -m uvicorn aquillm.asgi:application --host 0.0.0.0 --port ${PORT:-8080}
