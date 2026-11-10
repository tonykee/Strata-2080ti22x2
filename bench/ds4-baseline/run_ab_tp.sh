#!/bin/sh
# A/B/A/B of DS4_TP_SHARES in one boot: the same 13K prompt, 128 tokens, alternating.
#   bench/ds4-baseline/run_ab_tp.sh 4,5:base 3,6:tp36
set -eu
ROOT=/home/likan/strata-latest
cd "$ROOT"
A=${1:-clean-tp45}
B=${2:-clean-tp36}
SA=${3:-4,5}
SB=${4:-3,6}
for round in 1 2; do
    ./bench/ds4-baseline/run_one.sh "$A-r$round" "DS4_TP_SHARES=$SA" >/dev/null 2>&1 || true
    ./bench/ds4-baseline/run_one.sh "$B-r$round" "DS4_TP_SHARES=$SB" >/dev/null 2>&1 || true
done
for f in "$A-r1" "$B-r1" "$A-r2" "$B-r2"; do
    printf '%-14s ' "$f"
    grep -E '^prompt: ' "bench/ds4-baseline/runs/$f.log" | sed 's/.*(\(.*\) tok\/s.*/prefill \1/' | tr -d '\n'
    printf '  '
    grep -E '^decode: ' "bench/ds4-baseline/runs/$f.log" | sed 's/.*(\(.*\) tok\/s.*/decode \1/' | tr -d '\n'
    printf '  '
    grep -E '^decode per token' "bench/ds4-baseline/runs/$f.log" | sed 's/decode per token: //' | cut -c1-90
    printf '     experts in VRAM: '
    grep -oE 'experts [0-9.]+ GB \([0-9]+ of' "bench/ds4-baseline/runs/$f.log" | head -2 | tr '\n' ' '
    printf '\n'
done
