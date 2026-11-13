#!/bin/sh
# Stop the GSQ-RCO DeepSeek server (start_ds4_gsq_server.sh) and return its 2 MB huge-page pool.
#   ./stop_ds4_gsq.sh            # stop the server/engine, then release the pool (the RAM goes back)
#   ./stop_ds4_gsq.sh --keep     # stop, keep the pool reserved for the next start
cd /home/likan/strata-latest || exit 1

KEEP=0
for a in "$@"; do
    [ "$a" = "--keep" ] && KEEP=1
done

./stop_cpu_lane.sh
rc=$?

if [ "$KEEP" -eq 0 ]; then
    TOTAL=$(awk '/^HugePages_Total:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
    if [ "${TOTAL:-0}" -gt 0 ]; then
        echo "releasing the 2 MB pool (${TOTAL} pages)..."
        if ! sudo -n /usr/local/sbin/strata-ds4-hugepages-release; then
            echo "warning: the pool release failed (sudo rule?); it stays reserved, run it by hand:" >&2
            echo "  sudo /usr/local/sbin/strata-ds4-hugepages-release" >&2
        fi
    fi
else
    echo "2 MB pool kept (--keep); ./stop_ds4_gsq.sh without it returns the RAM."
fi
exit $rc
