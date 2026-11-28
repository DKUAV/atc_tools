#!/usr/bin/env bash
set -euo pipefail
case "${1:-}:$#" in
    simulate:1) export ATC_SIMULATE_COMPILE=1 ;;
    compile:3) export ATC_SIMULATE_COMPILE=0 ATC_AMCT_TARGETS=310B4,910B2 ;;
    *) echo 'Usage: service-smoke.sh simulate | compile MODEL REPORTS' >&2; exit 2 ;;
esac
mode=$1
shift
: "${ATC_APP_IMAGE:?Set the tested application image}"
smoke_dir=$(mktemp -d "${RUNNER_TEMP:-/tmp}/atc-smoke.XXXXXXXX")
export ATC_HOST_DATA_DIR="$smoke_dir/data" SSH_CONFIG_DIR="$smoke_dir/ssh"
export ATC_SMOKE_URL="http://127.0.0.1:${ATC_API_PORT:-8766}"
export ATC_BROWSER_SECURE_COOKIES=0
compose=(docker compose -p "atc-ci-$$" -f "${ATC_COMPOSE_FILE:-deploy/compose.yaml}")
cleanup() {
    local status=$?
    if (( status != 0 )); then "${compose[@]}" logs --no-color >&2 || true; fi
    "${compose[@]}" down || true
    sudo rm -rf -- "$smoke_dir"
    exit "$status"
}
trap cleanup EXIT
ssh-keygen -q -t ed25519 -N '' -f "$smoke_dir/key"
sudo install -d -m 0700 "$SSH_CONFIG_DIR"
sudo install -m 0600 "$smoke_dir/key.pub" "$SSH_CONFIG_DIR/authorized_keys"
"${compose[@]}" up -d --no-build --pull never --wait --wait-timeout 90
ssh_command=(ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    -o "UserKnownHostsFile=$smoke_dir/hosts" -i "$smoke_dir/key" -p "${ATC_SSH_PORT:-2222}" \
    root@127.0.0.1)
"${ssh_command[@]}" 'command -v atc && /opt/onnx/bin/python -I -m pip check'
app=("${compose[@]}" exec -T app)
export ATC_SMOKE_TOKEN_A
ATC_SMOKE_TOKEN_A=$("${app[@]}" atc_service issue-token smoke-a)
if [[ $mode == compile ]]; then
    uv run --locked python service/tests/compile_smoke.py "$@"
    exit 0
fi
export ATC_SMOKE_USER_A=smoke-a ATC_SMOKE_USER_B=smoke-b ATC_SMOKE_TOKEN_B
ATC_SMOKE_TOKEN_B=$("${app[@]}" atc_service issue-token smoke-b)
export ATC_SMOKE_PASSWORD_A ATC_SMOKE_PASSWORD_B
ATC_SMOKE_PASSWORD_A=$(openssl rand -hex 24)
ATC_SMOKE_PASSWORD_B=$(openssl rand -hex 24)
printf '%s\n' "$ATC_SMOKE_PASSWORD_A" "$ATC_SMOKE_PASSWORD_A" | "${app[@]}" atc_service set-browser-password smoke-a
printf '%s\n' "$ATC_SMOKE_PASSWORD_B" "$ATC_SMOKE_PASSWORD_B" | "${app[@]}" atc_service set-browser-password smoke-b
"${app[@]}" setpriv --reuid=20000 --regid=20000 --clear-groups -- sh -ec '
    test ! -w /data/service.sqlite3
    test ! -x /data/assets
    test ! -w /opt/onnx/bin/python
'
for suite in smoke browser_smoke delivery_smoke; do
    uv run --locked python "service/tests/$suite.py"
done
"${compose[@]}" exec -T --user 10001:10001 -e ATC_SMOKE_USER_A -e ATC_SMOKE_TOKEN_A app \
    /opt/onnx/bin/python -I - < service/tests/database_smoke.py
api_get() {
    curl --noproxy 127.0.0.1 -fsS -H "Authorization: Bearer $ATC_SMOKE_TOKEN_A" "$ATC_SMOKE_URL/v1/$1"
}
job_id=$(api_get jobs | jq -er 'first(.[] | select(.state=="succeeded" and .preset=="310b4-simulation")).id')
job=$(api_get "jobs/$job_id")
attempt=$(jq -er .attempt_id <<<"$job")
asset=$(jq -er 'first(.outputs[] | select(.kind=="simulated_om")).id' <<<"$job")
package=$("${app[@]}" /opt/onnx/bin/python -I -c 'import atc_tools; print(atc_tools.__file__)')
"${app[@]}" setpriv --reuid=20000 --regid=20000 --clear-groups -- sh -ec '
    test ! -w "$1"
    test ! -w "$(dirname "$1")"
    test ! -x "/data/runs/$2"
    test ! -w "/data/assets/$3"
' sh "$package" "$attempt" "$asset"
log_hash=$(api_get "jobs/$job_id/logs" | jq -er .text | sha256sum)
asset_hash=$(api_get "assets/$asset/download" | sha256sum)
if [[ -n ${ATC_SMOKE_WHEEL_DIR:-} ]]; then
    tar -C "$ATC_SMOKE_WHEEL_DIR" -cf - . | "${ssh_command[@]}" 'set -eu
        staging=$(mktemp -d)
        trap '\''rm -rf -- "$staging"'\'' EXIT
        cd "$staging"
        tar -xf -
        sha256sum -c SHA256SUMS
        /opt/onnx/bin/python -I -m pip install --no-index --no-deps --force-reinstall ./*.whl
        /opt/onnx/bin/python -I -m pip check'
    uv run --locked python service/tests/smoke.py
fi
"${compose[@]}" up -d --force-recreate --no-build --pull never --wait
test "$job" = "$(api_get "jobs/$job_id")"
test "$log_hash" = "$(api_get "jobs/$job_id/logs" | jq -er .text | sha256sum)"
test "$asset_hash" = "$(api_get "assets/$asset/download" | sha256sum)"
