#!/bin/sh
# Install the two huge-page helpers as root-owned files and allow likan to run exactly those
# two commands without a password.  Run with sudo; this does not grant general sudo.
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "run as root: sudo $0" >&2
    exit 1
fi

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
install -o root -g root -m 0755 "$HERE/ds4_hugepages.sh" /usr/local/sbin/strata-ds4-hugepages
install -o root -g root -m 0755 "$HERE/ds4_hugepages_release.sh" /usr/local/sbin/strata-ds4-hugepages-release
install -o root -g root -m 0440 "$HERE/sudoers-strata-ds4" /etc/sudoers.d/strata-ds4

if ! visudo -cf /etc/sudoers.d/strata-ds4; then
    rm -f /etc/sudoers.d/strata-ds4
    echo "sudoers validation failed; removed the rule." >&2
    exit 1
fi

echo "installed:"
echo "  /usr/local/sbin/strata-ds4-hugepages"
echo "  /usr/local/sbin/strata-ds4-hugepages-release"
echo "  /etc/sudoers.d/strata-ds4"
echo "likan may reserve 96 GB and release the pool without a password; nothing else is granted."
