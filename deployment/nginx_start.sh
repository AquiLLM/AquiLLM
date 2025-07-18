#!/bin/bash

envsubst '$HOST_NAME' < /tmp/nginx.conf | tee /etc/nginx/nginx.conf
exec /docker-entrypoint.sh
