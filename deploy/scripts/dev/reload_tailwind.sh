#!/bin/bash

set -e

cd /app/react
TEMPLATES_DIR=/app/aquillm/templates
COMPONENTS_DIR=/app/react/src

inotifywait -r -m "$TEMPLATES_DIR" "$COMPONENTS_DIR" -e modify |
while read -r directory event filename; do
  echo "Detected change in $filename, rebuilding Tailwind CSS..."
  npx tailwindcss -o /app/aquillm/aquillm/static/index.css
done