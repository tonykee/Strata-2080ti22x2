#!/bin/sh
# Stop start_cpu_lane.sh's server/engine and return its huge-page pool to ordinary memory.
# Only the release step needs sudo; the server runs as the invoking user.
set -u
cd /home/likan/strata-latest || exit 1

PIDFILE=.strata-ds4-v2.pid
FORCE=0
RELEASE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --release) RELEASE=1 ;;
    esac
done

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

# The engine is gone now.  By default the huge-page pool is KEPT: the next DeepSeek start
# reuses it (start_cpu_lane.sh reserves only when the pool is missing), so a stop/start of the
# server does not shuffle 96 GB of pages.  Measured on this PC: the first two engine starts
# after a reboot run at ~710 tok/s; later starts fall to ~560 until the next reboot, whether
# the pool is kept or re-reserved.  Use --release only when another model needs the RAM.
TOTAL=$(awk '/^HugePages_Total:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$RELEASE" -eq 1 ]; then
    if [ "${TOTAL:-0}" -gt 0 ] || [ "${DS4_COMPACT_ON_STOP:-0}" = 1 ]; then
        if [ "$(id -u)" -eq 0 ]; then
            tools/ds4_hugepages_release.sh
        else
            echo "releasing/compacting host memory (sudo)..."
            REL=tools/ds4_hugepages_release.sh
            [ -x /usr/local/sbin/strata-ds4-hugepages-release ] && REL=/usr/local/sbin/strata-ds4-hugepages-release
            sudo "$REL" || echo "warning: huge-page release failed; run sudo $REL" >&2
        fi
    fi
else
    echo "huge-page pool kept (HugePages_Total=${TOTAL:-0}); pass --release when switching to another model."
fi

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':8000 '; then
    echo "note: port 8000 is still listening (another tree/server may own it)." >&2
fi
echo "stopped."
