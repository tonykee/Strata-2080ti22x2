#!/bin/sh
# One measured engine load on the fastdec (hybrid) config.
#   bench/ds4-baseline/run_one.sh TAG [extra env assignments...]
# Writes bench/ds4-baseline/runs/<tag>.log and samples nvidia-smi beside it.
set -eu

TAG=$1
shift
ROOT=/home/likan/strata-latest
OUT=$ROOT/bench/ds4-baseline/runs
mkdir -p "$OUT"
LOG=$OUT/$TAG.log
SMI=$OUT/$TAG.smi

IDS=$(cat "$ROOT/bench/ds4-baseline/prompt-13k.txt")
PACK=/mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-XS-hybrid/maya-ds4-hybrid
PROFILE=/mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-XS-v2/maya-ds4-v2/expert-profile.bin

(
    while :; do
        nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current,memory.used --format=csv,noheader
        echo "--- $(date +%s)"
        sleep 5
    done
) > "$SMI" 2>&1 &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

set +e
DS4_ENGRAM_CACHE_MB=2048 DS4_SETTLE_MS=2000 DS4_IQ1S_ROWS=1 DS4_IQ_OWN=1 \
DS4_HIP_PEER_AR=1 DS4_STORE_GB=0 DS4_COPY_HUGETLB=1 DS4_COPY_PIN=1 DS4_ATTN_V7=1 \
DS4_CACHE_SUB_N=12 DS4_CACHE_SUB_RISK_BUDGET=0.6 DS4_CACHE_SUB_NORM=reference \
DS4_FAT_ROWS=64 DS4_RECENT=0 DS4_LAND=0 DS4_ATTN_SLAB=2048 DS4_AR2_DECODE=1 \
DS4_STREAM_ZC=1 DS4_COPY_PUMP=1 DS4_SKIP_UNPICKED_ROWS=2048 DS4_TP_SHARES=4,5 \
DS4_DSPARK_CONF=0.3 DS4_CACHE_SUB_RATIO=0.1 DS4_ROLLBACK_BATCH=1 DS4_ATTN_SPLIT=1 \
env "$@" \
    "$ROOT/build-dsv41/strata-dsv41" --pack "$PACK" \
    --expert-profile "$PROFILE" --profile-readonly \
    --gpus 0,1 --tp 2 --max-context 32768 --reserve-mib 768 --cpu-threads 20 \
    --mmap-experts off --experts 4 --prefill-chunk 8192 --dspark on \
    --ids "$IDS" --n 128 > "$LOG" 2>&1
rc=$?

grep -E 'HugePages_(Total|Free)' /proc/meminfo > "$OUT/$TAG.hugepages" 2>&1 || true
echo "rc=$rc"
grep -E 'load|prompt:|decode:|dspark:|resident|copied' "$LOG" | head -40
exit 0
