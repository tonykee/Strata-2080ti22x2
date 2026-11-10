#!/bin/sh
# Start Swift 1.5 IQ3_S with a 2 MB huge-page expert arena, and release the pool on exit.
#
#   ./start_swift_2mb_cache.sh              # batch 2 (concurrency), 96 GB pool
#   ./start_swift_2mb_cache.sh --single     # batch off, fastest single stream
#   HUGEPAGE_GB=48 ./start_swift_2mb_cache.sh --batch2   # 48 GB is enough for Swift; needs sudo password
#   HUGEPAGE_GB=0  ./start_swift_2mb_cache.sh             # no reservation; engine uses THP/4 KB
#   ./start_swift_2mb_cache.sh --check      # reserve, print the pool, release it; no server
#
# The pool is reserved only when it is not already big enough.  A pool this script reserved is
# released on Ctrl+C / SIGTERM / normal exit; a pool that was already there (e.g. DS4's) is left alone.
# The server runs in the foreground in its own session; ./stop_swift_2mb_cache.sh can stop only this
# instance from another terminal, without touching another server in this tree.
set -u
cd "/home/likan/strata-latest" || exit 1

PORT=${STRATA_PORT:-8000}
CONFIG=strata-swift-2mb-cache-batch2.json
CHECK=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --single) CONFIG=strata-swift-2mb-cache.json; shift ;;
        --batch2) CONFIG=strata-swift-2mb-cache-batch2.json; shift ;;
        --check) CHECK=1; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
HUGEGB=${HUGEPAGE_GB:-96}
NEED_PAGES=23983                 # 46.84 GiB expert arena / 2 MiB
MARKER=.swift-2mb-cache.pool
OWN_POOL=0
SERVER_PID=

if [ "$(id -u)" -eq 0 ]; then
    echo "run as the normal user, not root; sudo is used only for the huge-page steps." >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi
if [ "$CHECK" -eq 0 ] && command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT is already in use; stop the other server first (./stop.sh or ./stop_swift_2mb_cache.sh)." >&2
    exit 1
fi
if [ "$CHECK" -eq 0 ] && pgrep -x strata >/dev/null 2>&1; then
    echo "another strata engine is running; stop it first (./stop.sh), then rerun this script." >&2
    exit 1
fi
if pgrep -x strata-dsv41 >/dev/null 2>&1 || pgrep -x maya-ds4 >/dev/null 2>&1; then
    echo "a DS4 engine is running and may hold the huge-page pool; stop it first." >&2
    exit 1
fi

release_pool() {
    allow_other_strata=${1:-0}
    if [ "$OWN_POOL" -eq 1 ]; then
        if pgrep -x strata-dsv41 >/dev/null 2>&1 || pgrep -x maya-ds4 >/dev/null 2>&1; then
            echo "a DS4 engine is running; leaving the 2 MB pool reserved." >&2
        elif [ "$allow_other_strata" -eq 0 ] && pgrep -x strata >/dev/null 2>&1; then
            echo "another strata engine is running; leaving the 2 MB pool reserved." >&2
        else
            echo "releasing the 2 MB huge-page pool..."
            sudo -n /usr/local/sbin/strata-ds4-hugepages-release 2>/dev/null || \
                sudo tools/ds4_hugepages_release.sh 2>/dev/null || true
        fi
    fi
}

free_pages=$(awk '/^HugePages_Free:/{print $2}' /proc/meminfo)
total_pages=$(awk '/^HugePages_Total:/{print $2}' /proc/meminfo)
free_pages=${free_pages:-0}
total_pages=${total_pages:-0}

if [ "$free_pages" -ge "$NEED_PAGES" ]; then
    echo "using the existing 2 MB pool ($free_pages free pages); it will not be released on exit."
elif [ "$total_pages" -ge "$NEED_PAGES" ]; then
    echo "the 2 MB pool exists but only $free_pages pages are free (need $NEED_PAGES);" >&2
    echo "another process is using it. Stop that process, or lower HUGEPAGE_GB and retry." >&2
    exit 1
elif [ "$HUGEGB" -le 0 ] 2>/dev/null; then
    echo "2 MB pool is short ($free_pages free); HUGEPAGE_GB=$HUGEGB, using THP/4 KB (no prefill gain)." >&2
else
    echo "reserving ${HUGEGB} GB of 2 MB pages..."
    HUGE_CMD=/usr/local/sbin/strata-ds4-hugepages
    [ -x "$HUGE_CMD" ] || HUGE_CMD=tools/ds4_hugepages.sh
    if [ "$HUGEGB" -eq 96 ] && sudo -n "$HUGE_CMD" 96; then
        OWN_POOL=1
    elif sudo "$HUGE_CMD" "$HUGEGB"; then
        OWN_POOL=1
    else
        echo "huge-page reservation failed; rerun with HUGEPAGE_GB=0 to use THP/4 KB." >&2
        exit 1
    fi
    free_pages=$(awk '/^HugePages_Free:/{print $2}' /proc/meminfo)
    if [ "${free_pages:-0}" -lt "$NEED_PAGES" ]; then
        echo "the pool is still short ($free_pages < $NEED_PAGES); the arena would fall back." >&2
        release_pool
        exit 1
    fi
fi

if [ "$CHECK" -eq 1 ]; then
    echo "check: HugePages_Free=$(awk '/^HugePages_Free:/{print $2}' /proc/meminfo), need $NEED_PAGES; config $CONFIG"
    release_pool 1
    exit 0
fi

kill_server() {
    [ -n "$SERVER_PID" ] || return 0
    CHILDREN=$(pgrep -P "$SERVER_PID" 2>/dev/null || true)
    kill -TERM "$SERVER_PID" 2>/dev/null
    i=0
    while [ "$i" -lt 30 ] && kill -0 "$SERVER_PID" 2>/dev/null; do sleep 1; i=$((i + 1)); done
    kill -KILL "$SERVER_PID" 2>/dev/null
    for p in $CHILDREN; do kill -TERM "$p" 2>/dev/null; done
    sleep 1
    for p in $CHILDREN; do kill -KILL "$p" 2>/dev/null; done
}

cleanup() {
    trap - EXIT INT TERM HUP
    kill_server
    release_pool 0
    rm -f "$MARKER"
}
trap cleanup EXIT INT TERM HUP

PY=${STRATA_PYTHON:-/home/likan/strata/.venv/bin/python}
[ -x "$PY" ] || PY=python3
echo "starting $CONFIG on port $PORT; log: $(python3 -c "import json;print(json.load(open('$CONFIG'))['log'])" 2>/dev/null || echo 'the config log')"
echo "check for 'hugetlb 2 MB pages' in that log once the model is loaded."
"$PY" serve/server.py --engine strata --config "$CONFIG" --port "$PORT" "$@" &
SERVER_PID=$!
printf 'server_pid=%s\nown_pool=%s\nconfig=%s\n' "$SERVER_PID" "$OWN_POOL" "$CONFIG" > "$MARKER"
wait "$SERVER_PID"
