#!/bin/sh
# Defragment the host after a DS4 load has exited.  This is the command that matters for the
# CPU lane's ordinary THP arena: drop_caches only drops file pages, it does not coalesce free
# memory back into 2 MB blocks.  Run while no model process is loading/running.
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "run as root: sudo $0" >&2
    exit 1
fi

echo "before:"
grep -E 'AnonHugePages|HugePages_(Total|Free)' /proc/meminfo || true
awk '/Node/{print}' /proc/buddyinfo || true

sync
if [ -w /proc/sys/vm/compact_memory ]; then
    echo 1 > /proc/sys/vm/compact_memory
else
    echo "/proc/sys/vm/compact_memory is not writable" >&2
    exit 1
fi

echo "after:"
grep -E 'AnonHugePages|HugePages_(Total|Free)' /proc/meminfo || true
awk '/Node/{print}' /proc/buddyinfo || true
