#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
case "$mode:$#" in
    simulate:1) export ATC_SIMULATE_COMPILE=1 ;;
    compile:3)
        export ATC_SIMULATE_COMPILE=0
        : "${ATC_COMPILE_IMAGE:?Set the 310B4 executor image}"
        : "${ATC_COMPILE_910B2_IMAGE:?Set the 910B2 executor image}"
        ;;
    *) printf 'Usage: bash %s simulate | compile MODEL REPORTS\n' "$0" >&2; exit 2 ;;
esac
shift

if [[ -z ${ATC_HOST_DATA_DIR:-} ]]; then
    export ATC_HOST_DATA_DIR
    ATC_HOST_DATA_DIR=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/atc-smoke.XXXXXXXX")
fi
export ATC_DB_PASSWORD
ATC_DB_PASSWORD=$(openssl rand -hex 24)
export ATC_API_BIND=127.0.0.1 ATC_BROWSER_SECURE_COOKIES=0
export ATC_SMOKE_URL="http://127.0.0.1:${ATC_API_PORT:-8766}"
export ATC_SMOKE_DATA_DIR=$ATC_HOST_DATA_DIR
compose=(docker compose -p "atc-ci-${GITHUB_RUN_ID:-local}-$$" -f deploy/compose.yaml)

cleanup() {
    local status=$?
    if (( status != 0 )); then
        "${compose[@]}" logs --no-color >&2 || true
    fi
    "${compose[@]}" down --volumes || true
    exit "$status"
}
trap cleanup EXIT

if [[ ${GITHUB_ACTIONS:-} == true ]]; then
    printf '::add-mask::%s\n' "$ATC_DB_PASSWORD"
fi
"${compose[@]}" --profile build build api executor
"${compose[@]}" up -d --no-build --wait --wait-timeout 90
curl --fail --silent --show-error --retry 30 --retry-all-errors --retry-delay 1 \
    --max-time 5 "$ATC_SMOKE_URL/readyz"
printf '\n'

export ATC_SMOKE_TOKEN_A
ATC_SMOKE_TOKEN_A=$("${compose[@]}" exec -T api atc-service issue-token smoke-a)
if [[ ${GITHUB_ACTIONS:-} == true ]]; then
    printf '::add-mask::%s\n' "$ATC_SMOKE_TOKEN_A"
fi
if [[ $mode == compile ]]; then
    uv run --locked --no-dev python service/tests/compile_smoke.py "$@"
else
    export ATC_SMOKE_TOKEN_B ATC_SMOKE_USER_A=smoke-a ATC_SMOKE_USER_B=smoke-b
    export ATC_SMOKE_PASSWORD_A ATC_SMOKE_PASSWORD_B
    ATC_SMOKE_TOKEN_B=$("${compose[@]}" exec -T api atc-service issue-token smoke-b)
    ATC_SMOKE_PASSWORD_A=$(openssl rand -hex 24)
    ATC_SMOKE_PASSWORD_B=$(openssl rand -hex 24)
    if [[ ${GITHUB_ACTIONS:-} == true ]]; then
        printf '::add-mask::%s\n' "$ATC_SMOKE_TOKEN_B" "$ATC_SMOKE_PASSWORD_A" "$ATC_SMOKE_PASSWORD_B"
    fi
    printf '%s\n' "$ATC_SMOKE_PASSWORD_A" "$ATC_SMOKE_PASSWORD_A" \
        | "${compose[@]}" exec -T api atc-service set-browser-password "$ATC_SMOKE_USER_A"
    printf '%s\n' "$ATC_SMOKE_PASSWORD_B" "$ATC_SMOKE_PASSWORD_B" \
        | "${compose[@]}" exec -T api atc-service set-browser-password "$ATC_SMOKE_USER_B"
    for test in smoke browser_smoke delivery_smoke; do
        uv run --locked python "service/tests/$test.py"
    done
    "${compose[@]}" exec -T -e ATC_SMOKE_USER_A -e ATC_SMOKE_TOKEN_A api \
        python - < service/tests/database_smoke.py
fi
