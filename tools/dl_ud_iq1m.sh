#!/usr/bin/env bash
# Unsloth DeepSeek-V4.1-Flash-1017 UD-IQ1_M v3.1 (2026-10-28 revision) + the shared vision tower and
# DSpark drafter, resumable, ModelScope (魔塔) first and Hugging Face as the fallback.
#
#   ./tools/dl_ud_iq1m.sh                      # everything not yet complete
#   ./tools/dl_ud_iq1m.sh mmproj-BF16.gguf     # one file
#   ./tools/dl_ud_iq1m.sh --check              # sizes + sha256 only
#
# The Engram table is NOT downloaded: ModelScope's shard 5 is the same data as ISTA's
# DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf (setup.py says so and pack_gguf.py accepts it), so the copy in
# .../DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/ is reused.  Sizes and SHA-256 are setup.py's pins.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

DEST=${DEST:-/mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_M}
MS="https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/master"
HF="https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/8a6ec9cf118e785a362ea94ee123c61c70553be6"
JOBS=${JOBS:-2}

# name | ModelScope path | size | sha256
FILES=(
"DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf|UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf|5780868|5197c309479c7be06e3612dc696d7f18e13e5942cfe52e70c3c586b7e585a5ac"
"DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00002-of-00005.gguf|UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00002-of-00005.gguf|49707332128|fbfb546557a33f41332ffbed7f15fbed249b9f7796975d9db571c8cf3870f335"
"DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00003-of-00005.gguf|UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00003-of-00005.gguf|49292777280|394b7a06e04d4b1b96b6028ec3d4dec49e4716900c26a493e8c2aa38a357ea1a"
"DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00004-of-00005.gguf|UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00004-of-00005.gguf|40803165856|323a7f59a2938879eaac82b37c443c85151e518071ad44c79b1e4580c91d0ed7"
"mmproj-BF16.gguf|mmproj-BF16.gguf|972208288|9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9"
"dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf|dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf|7976914240|6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"
)

size_of() { stat -c %s "$DEST/$1" 2>/dev/null || echo 0; }
human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1"; }

check() {
    local bad=0
    echo "== check $DEST"
    for row in "${FILES[@]}"; do
        IFS='|' read -r n rel want sha <<<"$row"
        local got; got=$(size_of "$n")
        if [ "$got" -ne "$want" ]; then
            printf "  %-58s %s / %s  ✗\n" "$n" "$(human "$got")" "$(human "$want")"; bad=1; continue
        fi
        printf "  %-58s %s  size ✓" "$n" "$(human "$want")"
        if command -v sha256sum >/dev/null 2>&1; then
            local have; have=$(sha256sum "$DEST/$n" | cut -d' ' -f1)
            [ "$have" = "$sha" ] && echo "  sha256 ✓" || { echo "  sha256 ✗"; bad=1; }
        else echo; fi
    done
    [ "$bad" -eq 0 ] && echo "UD-IQ1_M files complete ✅" || echo "some files are missing or differ"
    return "$bad"
}

if [ "${1:-}" = "--check" ]; then check; exit $?; fi

mkdir -p "$DEST"
WANTED=("$@")
wanted() { [ ${#WANTED[@]} -eq 0 ] && return 0; for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

fetch_one() {
    local n=$1 rel=$2 want=$3 sha=$4
    local out="$DEST/$n" try=0
    while :; do
        try=$((try + 1))
        local have; have=$(size_of "$n")
        [ "$have" -ge "$want" ] && { echo "[完成] $n $(human "$have")"; return 0; }
        echo "[开始] $n 已有 $(human "$have") / $(human "$want") (第 $try 次)"
        curl -L --fail -C - \
             --retry 20 --retry-delay 10 --retry-all-errors \
             --speed-limit 1024 --speed-time 120 \
             --connect-timeout 30 -o "$out" "$MS/$rel"
        local rc=$?
        have=$(size_of "$n")
        if [ "$have" -lt "$want" ] && [ "$rc" -ne 0 ]; then
            echo "[回退] ModelScope 失败(rc=$rc)，改从 Hugging Face 续传"
            curl -L --fail -C - --retry 10 --retry-delay 10 --retry-all-errors \
                 --connect-timeout 30 -o "$out" "$HF/$n"
        fi
        have=$(size_of "$n")
        [ "$have" -ge "$want" ] && { echo "[完成] $n $(human "$have")"; return 0; }
        [ "$try" -ge 2000 ] && { echo "[放弃] $n 重试过多"; return 1; }
        sleep 10
    done
}
export -f fetch_one size_of human
export DEST MS HF

echo "== 目标目录: $DEST"
echo "== 起始: $(date '+%F %T')  并发: $JOBS  源: ModelScope（失败回退 HF）"
{
    for row in "${FILES[@]}"; do
        IFS='|' read -r n rel want sha <<<"$row"
        wanted "$n" || continue
        echo "$n|$rel|$want|$sha"
    done
} | sort -t'|' -k3 -nr | xargs -P "$JOBS" -d '\n' -I{} bash -c '
    IFS="|" read -r n rel want sha <<<"{}"
    fetch_one "$n" "$rel" "$want" "$sha"
'
echo "== 下载循环结束: $(date '+%F %T')"
check
