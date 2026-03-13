#!/bin/bash

set -e

normalize_openai_base_url() {
  local base="${1%/}"
  if [ -z "${base}" ]; then
    echo "http://vllm:8000/v1"
    return
  fi
  case "${base}" in
    */v1) echo "${base}" ;;
    *) echo "${base}/v1" ;;
  esac
}

configure_mem0() {
  if [ "${MEMORY_BACKEND:-local}" != "mem0" ]; then
    return 0
  fi

  case "$(echo "${MEM0_AUTO_CONFIGURE:-1}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) ;;
    *)
      echo "MEM0_AUTO_CONFIGURE disabled; skipping Mem0 /configure"
      return 0
      ;;
  esac

  if [ -z "${MEM0_BASE_URL:-}" ]; then
    echo "MEMORY_BACKEND=mem0 but MEM0_BASE_URL is not set; skipping Mem0 /configure"
    return 0
  fi

  local base_url="${MEM0_BASE_URL%/}"
  local llm_provider="${MEM0_LLM_PROVIDER:-openai}"
  local llm_model="${MEM0_LLM_MODEL:-qwen3.5:4b-q8_0}"
  local llm_base_url_raw="${MEM0_LLM_BASE_URL:-${MEM0_VLLM_BASE_URL:-${MEM0_OLLAMA_BASE_URL:-http://vllm:8000/v1}}}"
  local llm_base_url
  llm_base_url="$(normalize_openai_base_url "${llm_base_url_raw}")"
  local llm_api_key="${MEM0_LLM_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
  local embed_provider="${MEM0_EMBED_PROVIDER:-openai}"
  local embed_model="${MEM0_EMBED_MODEL:-nomic-ai/nomic-embed-text-v1.5}"
  local embed_base_url_raw="${MEM0_EMBED_BASE_URL:-${MEM0_VLLM_BASE_URL:-${MEM0_OLLAMA_BASE_URL:-http://vllm:8000/v1}}}"
  local embed_base_url
  embed_base_url="$(normalize_openai_base_url "${embed_base_url_raw}")"
  local embed_api_key="${MEM0_EMBED_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
  local qdrant_host="${MEM0_QDRANT_HOST:-qdrant}"
  local qdrant_port="${MEM0_QDRANT_PORT:-6333}"
  local collection_name="${MEM0_COLLECTION_NAME:-mem0_768_v4}"
  local embed_dims="${MEM0_EMBED_DIMS:-768}"

  echo "Waiting for Mem0 API at ${base_url}..."
  for i in $(seq 1 30); do
    if curl -fsS --max-time 2 "${base_url}/docs" >/dev/null 2>&1 || curl -fsS --max-time 2 "${base_url}/openapi.json" >/dev/null 2>&1; then
      break
    fi
    if [ "$i" -eq 30 ]; then
      echo "Mem0 API not reachable after 30 attempts; continuing without auto-configure"
      return 0
    fi
    sleep 2
  done

  local payload
  payload=$(cat <<JSON
{
  "version": "v1.1",
  "llm": {
    "provider": "${llm_provider}",
    "config": {
      "model": "${llm_model}",
      "openai_base_url": "${llm_base_url}",
      "api_key": "${llm_api_key}",
      "temperature": 0
    }
  },
  "embedder": {
    "provider": "${embed_provider}",
    "config": {
      "model": "${embed_model}",
      "openai_base_url": "${embed_base_url}",
      "api_key": "${embed_api_key}"
    }
  },
  "vector_store": {
    "provider": "qdrant",
    "config": {
      "host": "${qdrant_host}",
      "port": ${qdrant_port},
      "collection_name": "${collection_name}",
      "embedding_model_dims": ${embed_dims}
    }
  }
}
JSON
)

  if curl -fsS --max-time 30 -X POST "${base_url}/configure" -H 'Content-Type: application/json' -d "${payload}" >/dev/null; then
    echo "Mem0 configured successfully"
  else
    echo "Mem0 configure request failed; continuing startup"
  fi
}

cd /app/react
npm ci
npm run watch &
npx tailwindcss -o /app/aquillm/aquillm/static/index.css
npx tailwindcss -o /app/aquillm/aquillm/static/index.css
# I have no idea why it only works if you run it twice

/app/dev/reload_tailwind.sh &

cd /app/aquillm
./manage.py migrate --noinput
./manage.py collectstatic --noinput

configure_mem0

celery -A aquillm worker --loglevel=info &
python -Xfrozen_modules=off manage.py runserver 0.0.0.0:${PORT:-8080}
