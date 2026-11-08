#!/bin/sh
# Release the huge-page pool reserved for the DS4 CPU-lane arena and compact the host.
# Run after the engine has exited.  This is what a scoped service's ExecStopPost runs, so the
# next operation (DS4 again, Qwen, or anything else) sees vm.nr_hugepages=0 without a reboot.
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "run as root: sudo $0" >&2
    exit 1
fi

sync
sysctl -w vm.nr_hugepages=0 >/dev/null
if [ -w /proc/sys/vm/compact_memory ]; then
    echo 1 > /proc/sys/vm/compact_memory || true
fi
awk '/^HugePages_(Total|Free):|^Hugepagesize:/' /proc/meminfo
