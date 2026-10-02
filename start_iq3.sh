#!/bin/sh
# Primary deployment: v0.1.34 + LOCAL --stage-weights, 2x RTX 2080 Ti 22G (sm_75), port 8000.
# Explicit layer split 24 (--stage-weights needs it), prefill auto, kv-resident 32768.
cd "/home/likan/strata"
exec "/home/likan/strata/.venv/bin/python" "/home/likan/strata/serve/server.py" \
    "--engine" "strata" "--config" "/home/likan/strata/strata-swift-iq3_xxs.json" \
    "--port" "8000" "$@"
