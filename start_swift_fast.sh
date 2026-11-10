#!/bin/sh
# Start the fast/adapt Swift config with NO huge-page pool (THP pages).
#
#   ./start_swift_fast.sh          start in the background; returns at once
#   ./start_swift_fast.sh --wait   wait for /health (load takes about a minute)
#   ./stop_swift_fast.sh           stop this instance
#
# Config: strata-swift-fast.json (batch 2, --adapt-every 2, no vision, 393216/rope 1.5,
# STRATA_ARENA_HUGE_1G=1 but no pool: the engine tries 1 GB, then 2 MB, then THP).
# The script refuses to start if port 8000 or another strata engine is in use, and releases
# a leftover 2 MB/1 GB pool first so the run is really "no pool".
set -u
cd "/home/likan/strata-latest" || exit 1

PORT=${STRATA_PORT:-8000}
CONFIG=strata-swift-fast.json
PIDFILE=.strata-swift-fast.pid
LOG=${STRATA_LOG:-./strata-swift-fast-server.log}
WAIT=0
[ "${1:-}" = "--wait" ] && WAIT=1

if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi
if [ -f "$PIDFILE" ]; then
    OLD=$(cat "$PIDFILE" 2>/dev/null || true)
    if [ -n "${OLD:-}" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "already running: PID $OLD ($PIDFILE); use ./stop_swift_fast.sh first." >&2
        exit 1
    fi
    rm -f "$PIDFILE"
fi
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT is already in use; stop the other server first." >&2
    exit 1
fi
if pgrep -x strata >/dev/null 2>&1 || pgrep -x strata-dsv41 >/dev/null 2>&1 || pgrep -x maya-ds4 >/dev/null 2>&1; then
    echo "another strata/DS4 engine is running; stop it first (./stop.sh or ./stop_swift_fast.sh)." >&2
    exit 1
fi

# This config is meant to run without large pages: make sure a leftover pool is not picked up.
TOTAL=$(awk '/^HugePages_Total:/{print $2}' /proc/meminfo)
TOTAL=${TOTAL:-0}
if [ "$TOTAL" -gt 0 ]; then
    echo "releasing the leftover huge-page pool (this config runs on THP)..."
    if sudo -n /usr/local/sbin/strata-ds4-hugepages-release 2>/dev/null; then
        TOTAL=$(awk '/^HugePages_Total:/{print $2}' /proc/meminfo)
        TOTAL=${TOTAL:-0}
    fi
    if [ "$TOTAL" -gt 0 ]; then
        echo "could not release the huge-page pool (HugePages_Total=$TOTAL);" >&2
        echo "run: sudo /usr/local/sbin/strata-ds4-hugepages-release" >&2
        exit 1
    fi
fi

nohup /usr/bin/python3 /home/likan/strata-latest/strata.py serve \
    --profile "$CONFIG" --host 127.0.0.1 --port "$PORT" --quiet \
    > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
    echo "server did not stay up; last log lines:" >&2
    tail -20 "$LOG" >&2 2>/dev/null || true
    rm -f "$PIDFILE"
    exit 1
fi
echo "started PID $PID on port $PORT; log: $LOG"
echo "engine log: $(python3 -c "import json;print(json.load(open('$CONFIG'))['log'])" 2>/dev/null || echo 'the config log')"
if [ "$WAIT" -eq 1 ]; then
    echo "waiting for /health ..."
    for i in $(seq 1 180); do
        if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "health after $((i * 2))s"
            exit 0
        fi
        sleep 2
    done
    echo "health did not come up in 360 s; check $LOG" >&2
    exit 1
fi
