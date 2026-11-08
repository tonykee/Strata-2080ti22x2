#!/bin/sh
# DeepSeek-V4.1-Flash on 2x RTX 2080 Ti 22G (TP2, sm_75).
# This is a launcher, not a server by itself: pass --serve for the HTTP API, or --ids ... --n ... for a CLI run.
#
# Defaults are the measured 2x2080Ti configuration from DS4-MAYA-MERGE.md:
#   TP2, 4 experts, 8192-token prompt chunks, Engram cache 2048 MB, IQ1_S CPU rows.
#
# Optional knobs:
#   DS4_HIP_PEER_AR=0|1     CUDA peer prefill all-reduce (default 1; falls back to host if unavailable)
#   DS4_CACHE_SUB_RATIO=F   approximate cache substitution (default 0; 0.1 changes answers, opt-in)
#   DS4_IQ_OWN=0|1          force native / own IQ1_S rows (default 1, the checked-own form)
#   DS4_MMAP_EXPERTS=on|off (default off: 90 GB CPU-lane copy into RAM; on reads experts in place)
#   Before repeated reloads, reserve the huge-page pool once:
#     sudo tools/ds4_hugepages.sh 96 --persist
#   This keeps the CPU-lane arena on 2 MB pages and avoids the fragmentation/reboot cycle.
#   DS4_EXPERTS=N           experts per token (4 measured; 6 full quality, slower)
#   DS4_TP/N, DS4_GPUS, DS4_MAX_CONTEXT, DS4_RESERVE_MIB, DS4_PREFILL_CHUNK
cd /home/likan/strata-latest || exit 1

BIN=${STRATA_DS4_BIN:-./build-dsv41/strata-dsv41}
PACK=${DS4_PACK:-/mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-XS-v2/maya-ds4-v2}

if [ ! -x "$BIN" ]; then
    echo "Strata DS4 engine not found at $BIN; build build-dsv41/strata-dsv41 first." >&2
    exit 1
fi
if [ ! -d "$PACK" ]; then
    echo "DS4 pack not found at $PACK; set DS4_PACK." >&2
    exit 1
fi

export DS4_ENGRAM_CACHE_MB=${DS4_ENGRAM_CACHE_MB:-2048}
export DS4_SETTLE_MS=${DS4_SETTLE_MS:-2000}
export DS4_STREAM_ZC=${DS4_STREAM_ZC:-1}
export DS4_STREAM_WIN=${DS4_STREAM_WIN:-1}
export DS4_LAND=${DS4_LAND:-1}
export DS4_IQ1S_ROWS=${DS4_IQ1S_ROWS:-1}
export DS4_IQ_OWN=${DS4_IQ_OWN:-1}
export DS4_HIP_PEER_AR=${DS4_HIP_PEER_AR:-1}

# Cache substitution is approximate and off by default. Keep its companion values here so they
# travel with it when it is enabled explicitly.
export DS4_CACHE_SUB_RATIO=${DS4_CACHE_SUB_RATIO:-0}
export DS4_CACHE_SUB_N=${DS4_CACHE_SUB_N:-12}
export DS4_CACHE_SUB_RISK_BUDGET=${DS4_CACHE_SUB_RISK_BUDGET:-0.6}
export DS4_CACHE_SUB_NORM=${DS4_CACHE_SUB_NORM:-reference}

exec "$BIN" \
    --pack "$PACK" \
    --gpus "${DS4_GPUS:-0,1}" \
    --tp "${DS4_TP:-2}" \
    --max-context "${DS4_MAX_CONTEXT:-262144}" \
    --reserve-mib "${DS4_RESERVE_MIB:-768}" \
    --cpu-threads "${DS4_CPU_THREADS:-20}" \
    --mmap-experts "${DS4_MMAP_EXPERTS:-off}" \
    --experts "${DS4_EXPERTS:-4}" \
    --prefill-chunk "${DS4_PREFILL_CHUNK:-8192}" \
    --dspark "${DS4_DSPARK:-on}" \
    "$@"
