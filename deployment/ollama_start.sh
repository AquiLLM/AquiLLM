#!/bin/bash

set -e

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!


echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done

MODEL_TO_PULL=""
case "${LLM_CHOICE}" in
  GEMMA3)
    MODEL_TO_PULL="ebdm/gemma3-enhanced:12b"
    ;;
  LLAMA3.2)
    MODEL_TO_PULL="llama3.2"
    ;;
  GPT-OSS)
    MODEL_TO_PULL="gpt-oss:120b"
    ;;
  QWEN3_30B)
    MODEL_TO_PULL="qwen3:30b-a3b-instruct-2507-q4_K_M"
    ;;
  "")
    echo "LLM_CHOICE not set, skipping model pull."
    ;;
  *)
    echo "LLM_CHOICE '${LLM_CHOICE}' not recognized, skipping model pull."
    ;;
esac

if [ -n "${MODEL_TO_PULL}" ]; then
  echo "Pulling model: ${MODEL_TO_PULL}"
  ollama pull "${MODEL_TO_PULL}"
fi

echo "Ollama ready. Tailing server..."
wait $OLLAMA_PID