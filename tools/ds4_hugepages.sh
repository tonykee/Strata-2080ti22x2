#!/bin/sh
# Reserve Linux 2 MB huge pages for the DeepSeek-V4.1 CPU-lane copy.
#
# Why: with --mmap-experts off the engine copies ~91 GB of experts into an anonymous
# MADV_HUGEPAGE arena.  Repeated load/free cycles can leave the host fragmented: THP
# allocation then falls back to 4 KB pages and the CPU lane drops from ~10-15 ms/token
# to 40-90 ms/token until a reboot.  The engine already has an explicit MAP_HUGETLB
# arena (DS4_COPY_HUGETLB); this script reserves the pool once.  Huge pages are returned
# to the pool on process exit and reused by the next model load, so no reboot is needed.
#
# Usage:
#   sudo tools/ds4_hugepages.sh [GB] [--persist]
#
# Default 96 GB: the v2 pack's CPU-lane copy is about 90.7 GB; leave ~28 GB of the
# 127 GB for the OS, GPU host buffers and page cache.  Use a smaller value only with
# --mmap-experts on / DS4_COPY_GB, or if the machine needs more free RAM.
# --persist writes /etc/sysctl.d/99-strata-ds4-hugepages.conf so the reservation is
# restored at boot.  The engine then picks the huge arena automatically when the pool
# holds the whole copy (or set DS4_COPY_HUGETLB=1 explicitly).
set -eu

GB=96
PERSIST=0
for arg in "$@"; do
    case "$arg" in
        --persist) PERSIST=1 ;;
        *[!0-9]*|'') echo "usage: $0 [GB] [--persist]" >&2; exit 2 ;;
        *) GB=$arg ;;
    esac
done

if [ "$GB" -le 0 ]; then
    echo "GB must be positive" >&2
    exit 2
fi
if [ "$(id -u)" -ne 0 ]; then
    extra=
    [ "$PERSIST" -eq 1 ] && extra=" --persist"
    echo "run as root: sudo $0 $GB$extra" >&2
    exit 1
fi

SIZE_KB=$(awk '/^Hugepagesize:/{print $2}' /proc/meminfo)
[ -n "$SIZE_KB" ] || SIZE_KB=2048
PAGES=$((GB * 1024 * 1024 / SIZE_KB))
printf 'requesting %s GB as %s x %s KB huge pages\n' "$GB" "$PAGES" "$SIZE_KB"

# Compaction first: a fragmented host may otherwise allocate only part of the pool.
sync
if [ -w /proc/sys/vm/compact_memory ]; then
    echo 1 > /proc/sys/vm/compact_memory || true
fi
sysctl -w "vm.nr_hugepages=$PAGES" >/dev/null

TOTAL=$(awk '/^HugePages_Total:/{print $2}' /proc/meminfo)
FREE=$(awk '/^HugePages_Free:/{print $2}' /proc/meminfo)
printf 'HugePages_Total=%s  HugePages_Free=%s  (%.1f GB free)\n' \
    "$TOTAL" "$FREE" "$(awk -v f="$FREE" -v s="$SIZE_KB" 'BEGIN{printf "%.1f", f*s/1024/1024}')"

if [ "$TOTAL" -lt "$PAGES" ]; then
    echo "warning: the kernel reserved only $TOTAL of $PAGES pages; reboot with" >&2
    echo "  default_hugepagesz=2M hugepagesz=2M hugepages=$PAGES" >&2
    echo "on the kernel command line, or free/compact more memory and rerun." >&2
fi

if [ "$PERSIST" -eq 1 ]; then
    CONF=/etc/sysctl.d/99-strata-ds4-hugepages.conf
    printf '# Reserve 2 MB pages for the Strata DeepSeek CPU-lane arena.\nvm.nr_hugepages = %s\n' "$PAGES" > "$CONF"
    echo "persisted: $CONF"
fi

echo "launch with: DS4_COPY_HUGETLB=1 ./start_ds4_v2_2080ti.sh ..."
echo "verify after load: AnonHugePages should stay near 0 for the copy arena; HugePages_Free should fall by ~90.7 GB."
