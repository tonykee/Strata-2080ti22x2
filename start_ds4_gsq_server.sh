#!/bin/sh
# Start the GSQ-RCO DeepSeek-V4.1 server (OpenAI + Anthropic API, port 8000) on 2x RTX 2080 Ti.
# Reserves the 128 GB 2 MB huge-page pool first (passwordless helper), starts the server, then warms it.
#   ./start_ds4_gsq_server.sh                      # reserve 128 GB if needed, start, warm up
#   DS4_HUGEPAGE_GB=0 ./start_ds4_gsq_server.sh    # no reservation (ordinary pages)
#   STRATA_WARMUP=0 ./start_ds4_gsq_server.sh      # skip the ~107 s warm-up
# Stop with ./stop_ds4_gsq.sh (it returns the pool to the system) or `--keep` to hold it.
cd /home/likan/strata-latest || exit 1

# The 2 MB pool: reserve it when it does not already hold the asked size.  The rule in
# /etc/sudoers.d/strata-ds4-128 lets this run without a password; if it fails the engine starts on
# ordinary pages instead of stopping.  (DS4_COPY_HUGETLB is left unset: the engine takes the pool by
# itself when it holds the whole 122 GB copy.)
HUGEGB=${DS4_HUGEPAGE_GB:-128}
if [ "$HUGEGB" -gt 0 ] 2>/dev/null; then
    FREE_PAGES=$(awk '/^HugePages_Free:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
    WANT_PAGES=$((HUGEGB * 512))          # 512 x 2 MB pages a GB
    if [ "${FREE_PAGES:-0}" -lt "$WANT_PAGES" ]; then
        echo "reserving ${HUGEGB} GB of 2 MB pages (passwordless helper)..."
        if ! sudo -n /usr/local/sbin/strata-ds4-hugepages "$HUGEGB"; then
            echo "warning: the 2 MB pool could not be reserved; starting with ordinary pages" >&2
        fi
    else
        echo "2 MB pool: ${FREE_PAGES} pages already free ($((FREE_PAGES * 2048 / 1048576)) GB), reusing it"
    fi
fi

# The prompt lookup (the suffix drafter) runs without DSpark: on by default here.  DS4_LOOKUP_ALONE=0
# turns it off; a config with --suffix-draft 0 turns the lookup itself off.
DS4_LOOKUP_ALONE=${DS4_LOOKUP_ALONE:-1}
export DS4_LOOKUP_ALONE

# Reachable from the LAN (0.0.0.0) with the config's API key; STRATA_HOST=127.0.0.1 keeps it local.
DS4_CONFIG=${DS4_CONFIG:-./tools/strata-ds4-gsq-2080ti.json} \
DS4_HUGEPAGE_GB=0 \
DS4_LOG=${DS4_LOG:-./strata-ds4-gsq-server.log} \
STRATA_WARMUP=${STRATA_WARMUP:-1} \
    exec ./start_cpu_lane.sh --host "${STRATA_HOST:-0.0.0.0}" "$@"
