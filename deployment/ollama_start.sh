#!/bin/bash

set -e

echo "Starting Ollama server..."
ollama serve &


echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done

ollama pull ebdm/gemma3-enhanced:12b
ollama pull llama3.2
ollama pull gpt-oss:120b