#!/bin/bash

set -e
apt update
apt install docker-compose-v2 curl
export PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )
docker compose -f $PROJECT_ROOT/docker-compose-prod.yml run web ./manage.py migrate
envsubst '$PROJECT_ROOT' < "$PROJECT_ROOT/deployment/aquillm.service" > /etc/systemd/system/aquillm.service
systemctl enable aquillm.service 
systemctl start aquillm.service
