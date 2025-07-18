#!/bin/bash

set -e

if [ -z "${WM_EMAIL}" ] || [ -z "${HOST_NAME}" ]; then
    echo "Error: WM_EMAIL and HOST_NAME environment variables must be set"
    exit 1
fi

# Check if certificate directory already exists
if [ -d "/etc/letsencrypt/live/${HOST_NAME}" ]; then
    echo "Certificate directory already exists for ${HOST_NAME}"
    exit 0
fi

apt update
apt install python3 python3-venv libaugeas0 -y
python3 -m venv /opt/certbot/
/opt/certbot/bin/pip install certbot 
ln -s /opt/certbot/bin/certbot /usr/bin/certbot
certbot certonly --standalone --non-interactive --agree-tos -m ${WM_EMAIL} -d ${HOST_NAME} --cert-name ${HOST_NAME} -v
