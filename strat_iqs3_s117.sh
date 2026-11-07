#!/bin/sh
# Alias for the SC117 abliterated IQ3_S start script on the 0.2.11 branch.
# Canonical name: start_iq3s_117.sh (this wrapper keeps the older spelling working).
exec "$(dirname "$0")/start_iq3s_117.sh" "$@"
