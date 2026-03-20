#!/bin/bash

set -e

while true; do
    sleep 60s
    /usr/bin/curl -f http://localhost/health
done