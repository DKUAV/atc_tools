#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_shell.sh"

usage() {
    printf '%s\n' 'Usage: bash scripts/probe_env.sh [-e CANN_ENV] [-p PYTHON]' \
        '  -e, --cann-env   Source this CANN set_env.sh' \
        '  -p, --python     Python executable (default: python3)' \
        '  -h, --help       Show this help'
}

parsed=$(parse_options 'e:p:h' 'cann-env:,python:,help' "$@") || exit 2
eval "set -- $parsed"
env_file='' python=python3
while true; do
    case "$1" in
        -e|--cann-env) env_file=$2; shift 2 ;;
        -p|--python) python=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
    esac
done
[[ $# == 0 ]] || fail 'Unexpected positional arguments.'
load_cann "$env_file"

# Compile-only and runtime-only environments may expose different tools.
for tool in atc amct_onnx npu-smi "$python"; do
    printf '\n--- %s ---\n' "$tool"
    command -v "$tool" || printf 'Unavailable\n'
done
uname -sm
printf 'ASCEND_HOME_PATH=%s\nASCEND_OPP_PATH=%s\nASCEND_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_HOME_PATH:-}" "${ASCEND_OPP_PATH:-}" "${ASCEND_VISIBLE_DEVICES:-}"
if [[ -d ${ASCEND_OPP_PATH:-} ]]; then
    find "$ASCEND_OPP_PATH" -maxdepth 4 -name '*version*.info' -print
fi
if command -v atc >/dev/null; then atc --version || true; fi
if command -v npu-smi >/dev/null; then npu-smi info || true; fi
if command -v "$python" >/dev/null; then
    "$python" --version
    for module in acl numpy onnxruntime amct_onnx; do
        printf 'Import %s: ' "$module"
        "$python" -c 'import importlib, sys; print(importlib.import_module(sys.argv[1]).__file__)' "$module" || true
    done
fi
