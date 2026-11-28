#!/usr/bin/env bash
# SSH and CANN are initialized by the inherited image entrypoint.
set -euo pipefail
umask 077
data_dir=${ATC_DATA_DIR:-/data}
install -d -m 0711 -o 10001 -g 10001 "$data_dir" "$data_dir/runs"
install -d -m 0700 -o 10001 -g 10001 "$data_dir/assets" "$data_dir/uploads" "$data_dir/service-logs"
service=(setpriv --reuid=10001 --regid=10001 --clear-groups)
caps=+setuid,+setgid,+chown,+dac_override,+fowner,+kill
"${service[@]}" --inh-caps="$caps" --ambient-caps="$caps" -- atc-service scheduler &
scheduler=$!
"${service[@]}" --no-new-privs -- atc-service api &
api=$!
trap 'kill "$api" "$scheduler" 2>/dev/null || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
wait -n "$api" "$scheduler"
exit 1
