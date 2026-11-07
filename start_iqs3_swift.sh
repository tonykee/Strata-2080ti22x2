#!/bin/sh
# Swift 1.5 IQ3_S on the 0.2.11 / engine 0.1.57 branch (2x RTX 2080 Ti 22G, sm_75), port 8000.
# Config: strata-swift-latest.json (384K / yarn 1.5 / KV int8, batch 2, pipeline windows 2).
# Stop with ./stop.sh; do not put the literal engine/strata path on the same command line as ./stop.sh.
cd "/home/likan/strata-latest" || exit 1

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':8000 '; then
    echo "port 8000 is already in use - stop the other server first (production: /home/likan/strata/stop.sh, this branch: ./stop.sh)."
    exit 1
fi

exec "/home/likan/strata/.venv/bin/python" "/home/likan/strata-latest/serve/server.py" \
    "--engine" "strata" "--config" "/home/likan/strata-latest/strata-swift-latest.json" \
    "--port" "8000" "$@"
