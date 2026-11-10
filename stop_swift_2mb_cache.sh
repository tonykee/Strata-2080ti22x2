#!/bin/sh
# Stop the server started by start_swift_2mb_cache.sh and release the pool if that script reserved it.
# Use this when the start script was killed with SIGKILL (its EXIT trap did not run), or from another
# terminal.  It never touches another server in this tree: it uses the PID/session recorded by the
# start script.  A pool that was already reserved before the start script (e.g. DS4's) is left alone.
set -u
cd "/home/likan/strata-latest" || exit 1

MARKER=.swift-2mb-cache.pool
if [ ! -f "$MARKER" ]; then
    echo "no .swift-2mb-cache.pool marker: start_swift_2mb_cache.sh did not start a server here." >&2
    echo "for the branch's normal server use ./stop.sh; this script only handles the huge-page wrapper." >&2
    exit 0
fi

SERVER_PID=$(sed -n 's/^server_pid=//p' "$MARKER" | head -1)
OWN_POOL=$(sed -n 's/^own_pool=//p' "$MARKER" | head -1)
CONFIG=$(sed -n 's/^config=//p' "$MARKER" | head -1)

if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping server $SERVER_PID ($CONFIG)..."
    CHILDREN=$(pgrep -P "$SERVER_PID" 2>/dev/null || true)
    kill -TERM "$SERVER_PID" 2>/dev/null
    i=0
    while [ "$i" -lt 30 ] && kill -0 "$SERVER_PID" 2>/dev/null; do sleep 1; i=$((i + 1)); done
    kill -KILL "$SERVER_PID" 2>/dev/null
    for p in $CHILDREN; do kill -TERM "$p" 2>/dev/null; done
    sleep 1
    for p in $CHILDREN; do kill -KILL "$p" 2>/dev/null; done
else
    echo "server $SERVER_PID is not running (or no PID in the marker)."
fi

if [ "${OWN_POOL:-0}" = "1" ]; then
    if pgrep -x strata-dsv41 >/dev/null 2>&1 || pgrep -x maya-ds4 >/dev/null 2>&1 || pgrep -x strata >/dev/null 2>&1; then
        echo "another engine is running; leaving the 2 MB pool reserved." >&2
    else
        echo "releasing the 2 MB huge-page pool..."
        sudo -n /usr/local/sbin/strata-ds4-hugepages-release 2>/dev/null || \
            sudo tools/ds4_hugepages_release.sh 2>/dev/null || true
    fi
else
    echo "the pool was not reserved by the start script; leaving it as it is."
fi

rm -f "$MARKER"
awk '/^HugePages_(Total|Free):|^Hugepagesize:/' /proc/meminfo
