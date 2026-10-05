#!/bin/sh
# Primary deployment: v0.1.39 + #742/#743 (Turing long-prompt prefill), 2x RTX 2080 Ti 22G (sm_75), port 8000.
# layer split 24 + --trim-stage-weights, --batch 2 --batch-groups 2, parking, prefill auto, kv-resident 32768.
# Needs STRATA_BF16_TC=1 (v0.1.39 disables the BF16->FP16 prefill path on sm_75 by default).
cd "/home/likan/strata"
exec "/home/likan/strata/.venv/bin/python" "/home/likan/strata/serve/server.py" \
    "--engine" "strata" "--config" "/home/likan/strata/strata-swift-iq3_xxs.json" \
    "--port" "8000" "$@"
