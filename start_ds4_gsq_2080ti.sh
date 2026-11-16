#!/bin/sh
# DeepSeek-V4.1-Flash-1017 GSQ-RCO (IQ1_M / IQ2_XXS experts, 1.94 bpw) on 2x RTX 2080 Ti 22G.
# Same launcher shape as start_ds4_v2_2080ti.sh; this one points at the GSQ-RCO pack that Maya
# prepared at .../DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/maya-ds4-iq2xxs-dspark.
#
# The pack was built with the shared Engram table (110.8 GB, read from the SSD) and the DSpark
# drafter (4.25 GB), so --dspark on works here. Experts are 132.1 GB; VRAM holds about a sixth
# and the rest is copied into RAM (--mmap-experts off), which is why this host wants ~110 GB free.
#
# This is a launcher, not a server by itself: pass --serve for the HTTP API, or --ids ... --n ...
# for a CLI run.
#
# Optional knobs:
#   DS4_HIP_PEER_AR=0|1     CUDA peer prefill all-reduce (default 1; falls back to host if unavailable)
#   DS4_CACHE_SUB_RATIO=F   approximate cache substitution (default 0; 0.1 changes answers, opt-in)
#   DS4_MMAP_EXPERTS=on|off (default off: the CPU lane's experts copied into RAM; on reads in place)
#   DS4_EXPERTS=N           experts per token (4 measured; 6 full quality, slower)
#   DS4_TP/N, DS4_GPUS, DS4_MAX_CONTEXT, DS4_RESERVE_MIB, DS4_PREFILL_CHUNK
#   DS4_PROFILE=FILE|none   expert profile (default: the pack's own expert-profile.bin)
#   DS4_PACK=DIR            the pack (default: maya-ds4-iq2xxs-vision, pictures and no DSpark;
#                           ...-dspark keeps the drafter at ~1300 fewer VRAM experts)
#   DS4_SUFFIX_DRAFT=N      the prompt lookup's shortest match (default 3; 0 turns the lookup off);
#                           the lookup runs without DSpark, and DS4_LOOKUP_ALONE=0 (serve) turns its
#                           environment switch off
#   Before repeated reloads, reserve the 2 MB pool once (needs a password):
#     sudo tools/ds4_hugepages.sh 96 --persist
#   and export DS4_COPY_HUGETLB=1; without it the copy uses ordinary THP.
cd /home/likan/strata-latest || exit 1

BIN=${STRATA_DS4_BIN:-./build-dsv41/strata-dsv41}
PACK=${DS4_PACK:-/mnt/kioxia/Maya-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/maya-ds4-iq2xxs-vision}

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
# --mmap-experts off copies the CPU lane's experts into RAM. Pinning that copy is what lets the prompt
# path DMA them to the GPU (measured on this box: 35 tok/s prefill unpinned against 200+ pinned); the
# pump overlaps the copies. The RAM store would read the same experts a second time, so it is off.
export DS4_COPY_PIN=${DS4_COPY_PIN:-1}
export DS4_COPY_PUMP=${DS4_COPY_PUMP:-1}
export DS4_STORE_GB=${DS4_STORE_GB:-0}

# Cache substitution is approximate and off by default. Keep its companion values here so they
# travel with it when it is enabled explicitly.
export DS4_CACHE_SUB_RATIO=${DS4_CACHE_SUB_RATIO:-0}
export DS4_CACHE_SUB_N=${DS4_CACHE_SUB_N:-12}
export DS4_CACHE_SUB_RISK_BUDGET=${DS4_CACHE_SUB_RISK_BUDGET:-0.6}
export DS4_CACHE_SUB_NORM=${DS4_CACHE_SUB_NORM:-reference}

# The prompt lookup runs without DSpark; the CLI builds it for --suffix-draft > 0 (default 3), and
# --serve needs this environment switch.
export DS4_LOOKUP_ALONE=${DS4_LOOKUP_ALONE:-1}

# The drafter pays only where the cache substitution keeps the CPU lane small.  Measured 2026-11-13
# (same 215-token prompt, warm-up + 128-token answer, 3 runs): faithful (no substitution) 11.03 tok/s
# with DSpark on against 13.76 / 14.98 with it off; with substitution 0.1, 22.23 on against 17.35 off.
# So: on beside the substitution, off without it, unless the caller says which.
if [ -z "${DS4_DSPARK:-}" ]; then
    case "${DS4_CACHE_SUB_RATIO}" in
        0|0.0|0.00) DS4_DSPARK=off ;;
        *) DS4_DSPARK=on ;;
    esac
fi
export DS4_DSPARK

PROFILE_ARGS=""
case "${DS4_PROFILE:-auto}" in
    none) ;;
    auto)
        # A profile the engine may update beats the pack's read-only one: it learns the hot experts from
        # this host's own routing.  Measured 2026-11-13: 23.1 -> 28.5 tok/s warm decode after three runs.
        if [ -f "$PWD/ds4-gsq-expert-profile.bin" ]; then
            PROFILE_ARGS="--expert-profile $PWD/ds4-gsq-expert-profile.bin"
            DS4_PROFILE_RW=${DS4_PROFILE_RW:-1}
        elif [ -f "$PACK/expert-profile.bin" ]; then
            PROFILE_ARGS="--expert-profile $PACK/expert-profile.bin"
        fi ;;
    *)
        PROFILE_ARGS="--expert-profile $DS4_PROFILE"
        [ -f "$DS4_PROFILE" ] && DS4_PROFILE_RW=${DS4_PROFILE_RW:-1} ;;
esac
# Read-only unless the profile is one of ours: a writable profile is what lets a run's own routing update
# the hot-expert set.
[ -n "$PROFILE_ARGS" ] && [ -z "${DS4_PROFILE_RW:-}" ] && PROFILE_ARGS="$PROFILE_ARGS --profile-readonly"

# shellcheck disable=SC2086
exec "$BIN" \
    --pack "$PACK" \
    $PROFILE_ARGS \
    --gpus "${DS4_GPUS:-0,1}" \
    --tp "${DS4_TP:-2}" \
    --max-context "${DS4_MAX_CONTEXT:-393216}" \
    --reserve-mib "${DS4_RESERVE_MIB:-768}" \
    --cpu-threads "${DS4_CPU_THREADS:-10}" \
    --mmap-experts "${DS4_MMAP_EXPERTS:-off}" \
    --experts "${DS4_EXPERTS:-4}" \
    --prefill-chunk "${DS4_PREFILL_CHUNK:-8192}" \
    --dspark "${DS4_DSPARK:-on}" \
    --suffix-draft "${DS4_SUFFIX_DRAFT:-3}" \
    "$@"
