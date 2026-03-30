#!/bin/bash
set -e

if [ "${RUN_CELERY_IN_WEB:-1}" = "1" ]; then
  uv run celery -A aquillm worker --loglevel=info &
fi

cd /app/react
npm ci
npm run build
npx tailwindcss -o /app/aquillm/aquillm/static/index.css
npx tailwindcss -o /app/aquillm/aquillm/static/index.css
# I have no idea why it only works if you run it twice

cd /app/aquillm

uv run ./manage.py migrate --noinput
uv run ./manage.py collectstatic --noinput
exec uv run python -m uvicorn aquillm.asgi:application --host 0.0.0.0 --port ${PORT:-8080}
