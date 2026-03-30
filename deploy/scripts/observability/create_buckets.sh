#!/bin/bash
# Create all MinIO buckets needed by the observability stack and the app.
/usr/bin/mc alias set aquillm http://storage:9000 dev rickbailey
/usr/bin/mc mb --ignore-existing aquillm/aquillm
/usr/bin/mc mb --ignore-existing aquillm/tempo
/usr/bin/mc mb --ignore-existing aquillm/loki
/usr/bin/mc mb --ignore-existing aquillm/pyroscope
