#!/bin/sh
# SC117 abliterated IQ3_S on the 0.2.11 / engine 0.1.57 branch (2x RTX 2080 Ti 22G, sm_75), port 8000.
# Config: strata-sc117-latest.json. Stop with ./stop.sh.
cd "/home/likan/strata-latest" || exit 1

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':8000 '; then
    echo "port 8000 is already in use - stop the other server first (production: /home/likan/strata/stop.sh, this branch: ./stop.sh)."
    exit 1
fi

exec "/home/likan/strata/.venv/bin/python" "/home/likan/strata-latest/serve/server.py" \
    "--engine" "strata" "--config" "/home/likan/strata-latest/strata-sc117-latest.json" \
    "--port" "8000" "$@"
