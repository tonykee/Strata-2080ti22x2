#!/bin/sh
# Stop the server started by start_swift_fast.sh (PID file, no broad pattern matching).
set -u
cd "/home/likan/strata-latest" || exit 1

PIDFILE=.strata-swift-fast.pid
if [ ! -f "$PIDFILE" ]; then
    echo "no $PIDFILE: start_swift_fast.sh is not running here." >&2
    exit 0
fi
PID=$(cat "$PIDFILE" 2>/dev/null || true)
if [ -z "${PID:-}" ]; then
    echo "empty $PIDFILE; removing it." >&2
    rm -f "$PIDFILE"
    exit 0
fi
if kill -0 "$PID" 2>/dev/null; then
    echo "stopping PID $PID ..."
    CHILDREN=$(pgrep -P "$PID" 2>/dev/null || true)
    kill -TERM "$PID" 2>/dev/null
    i=0
    while [ "$i" -lt 30 ] && kill -0 "$PID" 2>/dev/null; do sleep 1; i=$((i + 1)); done
    kill -KILL "$PID" 2>/dev/null
    for p in $CHILDREN; do kill -TERM "$p" 2>/dev/null; done
    sleep 1
    for p in $CHILDREN; do kill -KILL "$p" 2>/dev/null; done
else
    echo "PID $PID is not running (stale PID file)."
fi
rm -f "$PIDFILE"
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':8000 '; then
    echo "WARNING: port 8000 is still listening (another server, or an orphan)." >&2
else
    echo "stopped; port 8000 free."
fi
