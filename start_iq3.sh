#!/bin/sh
# Independent v0.1.34 test deployment in ~/strata-v0134 (does NOT touch ~/strata).
# 2x RTX 2080 Ti 22G, sm_75.  Explicit layer split 24 (needed by --stage-weights), prefill auto.
cd "/home/likan/strata-v0134"
exec "/home/likan/strata/.venv/bin/python" "/home/likan/strata-v0134/serve/server.py" \
    "--engine" "strata" "--config" "/home/likan/strata-v0134/strata-swift-iq3_xxs.json" \
    "--port" "8000" "$@"
