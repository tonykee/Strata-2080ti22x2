#!/bin/sh
# Stop start_cpu_lane.sh's server/engine and return its huge-page pool to ordinary memory.
# Only the release step needs sudo; the server runs as the invoking user.
set -u
cd /home/likan/strata-latest || exit 1

PIDFILE=.strata-ds4-v2.pid
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

stop_pid() {
    pid=$1
    [ -n "$pid" ] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    echo "stopping PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
    [ "$FORCE" -eq 1 ] && return 0
    i=0
    while [ "$i" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        i=$((i + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "force-killing PID $pid"
        kill -KILL "$pid" 2>/dev/null || true
    fi
}

if [ -f "$PIDFILE" ]; then
    stop_pid "$(cat "$PIDFILE" 2>/dev/null || true)"
    rm -f "$PIDFILE"
else
    echo "$PIDFILE not found; looking for this tree's server/engine."
fi

# This tree only: do not touch production /home/likan/strata or the Qwen engine.
SERVER_PAT="[/]home/likan/strata-latest/serve/server.py.*strata-ds4-v2-2080ti.json"
ENGINE_PAT="[/]home/likan/strata-latest/build-dsv41/strata-dsv41 --serve"
if command -v pgrep >/dev/null 2>&1; then
    for pat in "$SERVER_PAT" "$ENGINE_PAT"; do
        PIDS=$(pgrep -f "$pat" 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            echo "stopping leftovers ($pat): $(echo "$PIDS" | tr '\n' ' ')"
            kill -TERM $PIDS 2>/dev/null || true
            if [ "$FORCE" -eq 0 ]; then
                i=0
                while [ "$i" -lt 10 ] && pgrep -f "$pat" >/dev/null 2>&1; do
                    sleep 1
                    i=$((i + 1))
                done
            fi
            PIDS=$(pgrep -f "$pat" 2>/dev/null || true)
            [ -n "$PIDS" ] && kill -KILL $PIDS 2>/dev/null || true
        fi
    done
fi

# The engine is gone now: return the huge-page pool to normal memory.  If no pool is
# reserved (for example the start used DS4_HUGEPAGE_GB=0), this needs no sudo at all.
TOTAL=$(awk '/^HugePages_Total:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
COMPACT=${DS4_COMPACT_ON_STOP:-0}
if [ "${TOTAL:-0}" -gt 0 ] || [ "$COMPACT" = 1 ]; then
    if [ "$(id -u)" -eq 0 ]; then
        tools/ds4_hugepages_release.sh
    else
        echo "releasing/compacting host memory (sudo)..."
        sudo tools/ds4_hugepages_release.sh || echo "warning: huge-page release failed; run sudo tools/ds4_hugepages_release.sh" >&2
    fi
else
    echo "no reserved huge-page pool; ordinary THP arena exits without sudo (set DS4_COMPACT_ON_STOP=1 to compact)."
fi

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':8000 '; then
    echo "note: port 8000 is still listening (another tree/server may own it)." >&2
fi
echo "stopped."
