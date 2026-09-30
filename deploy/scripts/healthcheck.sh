#!/bin/bash

set -eu
: "${HOST_NAME:?HOST_NAME must match the nginx virtual host}"

while true; do
    sleep 60s
    # Report outages without tearing down unrelated model and worker services.
    if ! /usr/bin/curl --fail --silent --show-error --connect-timeout 2 --max-time 5 \
        -H "Host: ${HOST_NAME}" http://localhost/health; then
        echo "AquiLLM HTTP health probe failed; containers retain their own restart policies" >&2
    fi
done
