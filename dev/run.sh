#!/bin/bash

set -e



cd /app/react
npm ci
npm run watch &
npx tailwindcss -o /app/aquillm/aquillm/static/index.css
npx tailwindcss -o /app/aquillm/aquillm/static/index.css
# I have no idea why it only works if you run it twice

/app/dev/reload_tailwind.sh &

cd /app/aquillm
uv run manage.py migrate --noinput
uv run manage.py collectstatic --noinput

celery -A aquillm worker --loglevel=info &
uv run manage.py runserver 0.0.0.0:${PORT:-8080}