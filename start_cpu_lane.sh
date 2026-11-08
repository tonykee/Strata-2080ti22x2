#!/bin/sh
# Start the DeepSeek-V4.1 CPU-lane server on 2x RTX 2080 Ti with a process-scoped huge-page pool.
#
# Run as the normal user (not with sudo): only the huge-page reservation uses sudo; the
# Python server and strata-dsv41 engine run as you.  Set DS4_HUGEPAGE_GB=0 to skip the
# reservation entirely (no sudo needed; the engine then uses its ordinary THP arena).
set -u
cd /home/likan/strata-latest || exit 1

PORT=${STRATA_PORT:-8000}
CONFIG=${DS4_CONFIG:-./tools/strata-ds4-v2-2080ti.json}
HUGEGB=${DS4_HUGEPAGE_GB:-96}
PIDFILE=.strata-ds4-v2.pid
LOG=${DS4_LOG:-./strata-ds4-v2-server.log}

if [ "$(id -u)" -eq 0 ]; then
    echo "run as the normal user, not root; the script uses sudo only for the huge-page step." >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi
if [ -f "$PIDFILE" ]; then
    OLD=$(cat "$PIDFILE" 2>/dev/null || true)
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "already running: PID $OLD ($PIDFILE); use ./stop_cpu_lane.sh first." >&2
        exit 1
    fi
    rm -f "$PIDFILE"
fi
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT is already in use; stop the other server first." >&2
    exit 1
fi

if [ "$HUGEGB" -gt 0 ] 2>/dev/null; then
    echo "reserving ${HUGEGB} GB of 2 MB huge pages (sudo)..."
    if ! sudo tools/ds4_hugepages.sh "$HUGEGB"; then
        echo "huge-page reservation failed; fix it or rerun with DS4_HUGEPAGE_GB=0." >&2
        exit 1
    fi
else
    echo "huge-page reservation skipped (DS4_HUGEPAGE_GB=$HUGEGB)."
fi

PY=${STRATA_PYTHON:-/home/likan/strata/.venv/bin/python}
[ -x "$PY" ] || PY=python3
nohup "$PY" serve/server.py --engine strata --config "$CONFIG" --port "$PORT" >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "server did not stay up; last log lines:" >&2
    tail -20 "$LOG" >&2 2>/dev/null || true
    rm -f "$PIDFILE"
    exit 1
fi
echo "started PID $PID on port $PORT; log: $LOG"
echo "engine load can take ~2-3 minutes; check: tail -f $LOG"
