#!/bin/sh
# Stop the 0.2.11 / engine 0.1.57 branch's server and its engine/vision children.
# Path-specific: it does NOT stop the production tree in /home/likan/strata.
#
#   ./stop.sh          graceful stop, then force-kill leftovers
#   ./stop.sh --force  skip the graceful wait
cd "/home/likan/strata-latest" || exit 1

FORCE=0
[ "$1" = "--force" ] && FORCE=1

# The leading [/] keeps pgrep/pkill from matching this script's own command line.
SERVER_PAT="[/]home/likan/strata-latest/serve/server.py"
ENGINE_PAT="[/]home/likan/strata-latest/engine/strata"
VISION_PAT="[/]home/likan/strata-latest/engine/strata-vision"

pids() { pgrep -f "$1" 2>/dev/null; }

SERVER_PIDS=$(pids "$SERVER_PAT")
if [ -z "$SERVER_PIDS" ]; then
    echo "v0.2.11 server is not running."
else
    echo "Stopping v0.2.11 server: $SERVER_PIDS"
    kill -TERM $SERVER_PIDS 2>/dev/null
    if [ "$FORCE" -eq 0 ]; then
        i=0
        while [ "$i" -lt 30 ]; do
            pids "$SERVER_PAT" >/dev/null 2>&1 || break
            sleep 1
            i=$((i + 1))
        done
    fi
fi

# Force-kill anything still alive (server or orphaned engine/vision children).
LEFT=$(pids "$SERVER_PAT"; pids "$VISION_PAT"; pids "$ENGINE_PAT")
if [ -n "$LEFT" ]; then
    echo "Force-killing leftovers: $(echo $LEFT | tr '\n' ' ')"
    kill -KILL $LEFT 2>/dev/null
    sleep 1
fi

LEFT=$(pids "$SERVER_PAT"; pids "$VISION_PAT"; pids "$ENGINE_PAT")
if [ -n "$LEFT" ]; then
    echo "WARNING: still running: $(echo $LEFT | tr '\n' ' ')"
    exit 1
fi

# Port check. Another tree (production) may legitimately hold 8000.
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':8000 '; then
    echo "WARNING: port 8000 is still listening (the production server, or another process)."
fi

echo "Stopped. GPUs/RAM freed."
