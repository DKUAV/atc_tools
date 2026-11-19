# Shared setup for Bash entry points.
set -euo pipefail

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

parse_options() {
    local short_options=$1 long_options=$2 status=0
    shift 2
    command -v getopt >/dev/null || fail 'util-linux getopt is required.'
    (unset GETOPT_COMPATIBLE POSIXLY_CORRECT; getopt -T) >/dev/null 2>&1 || status=$?
    [[ $status == 4 ]] || fail 'Enhanced util-linux getopt is required.'
    (unset GETOPT_COMPATIBLE POSIXLY_CORRECT
     getopt --shell bash --options "$short_options" --longoptions "$long_options" -- "$@")
}

load_cann() {
    local env_file=$1
    if [[ -n $env_file ]]; then
        [[ -f $env_file ]] || fail "CANN environment file not found: $env_file"
        # Vendor scripts may expand unset variables.
        set +u
        source "$env_file"
        set -u
    fi
}

run_logged() {
    local output=$1 statuses=()
    shift
    printf '%q ' "$@" > "$output/command.txt"
    printf '\n' >> "$output/command.txt"
    set +e
    "$@" 2>&1 | tee "$output/tool.log"
    statuses=("${PIPESTATUS[@]}")
    set -e
    printf '%s\n' "${statuses[0]}" > "$output/exit-code.txt"
    if [[ ${statuses[0]} != 0 ]]; then return "${statuses[0]}"; fi
    return "${statuses[1]}"
}
