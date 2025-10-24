#!/bin/bash

set -e

if [ -z "${WM_EMAIL}" ] || [ -z "${HOST_NAME}" ]; then
    echo "Error: WM_EMAIL and HOST_NAME environment variables must be set"
    exit 1
fi

# Check if certificate directory already exists
if [ -d "/etc/letsencrypt/live/${HOST_NAME}" ]; then
    echo "Certificate directory already exists for ${HOST_NAME}, attempting renewal"
    certbot renew --standalone --non-interactive -v
else
    echo "Obtaining new certificate for ${HOST_NAME}"
    certbot certonly --standalone --non-interactive --agree-tos -m ${WM_EMAIL} -d ${HOST_NAME} --cert-name ${HOST_NAME} -v
fi
