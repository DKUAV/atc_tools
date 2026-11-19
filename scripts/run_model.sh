#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_shell.sh"

usage() {
    printf '%s\n' 'Usage: bash scripts/run_model.sh -m OM -i NAME=FILE -o NEW_DIR [options]' \
        '  -m, --model          OM model' \
        '  -i, --input          NAME=FILE (.npy, .bin, .jpg, .jpeg, .png), repeatable' \
        '  -o, --output-dir     New directory for raw model outputs' \
        '  -d, --device         Logical device ID (default: 0)' \
        '  -r, --reference      Directory produced by prepare_model.py reference' \
        '  -a, --atol           Absolute tolerance (default: 0.001)' \
        '  -t, --rtol           Relative tolerance (default: 0.01)' \
        '  -n, --output-map     ONNX_NAME=OM_NAME override, repeatable' \
        '  -w, --warmup         Warmup runs (default: 0)' \
        '  -b, --repeat         Timed runs (default: 1)' \
        '  -p, --python         Python executable (default: python3)' \
        '  -e, --cann-env       CANN set_env.sh' \
        '  -h, --help           Show this help'
}

parsed=$(parse_options 'm:i:o:d:r:a:t:n:w:b:p:e:h' 'model:,input:,output-dir:,device:,reference:,atol:,rtol:,output-map:,warmup:,repeat:,python:,cann-env:,help' "$@") || exit 2
eval "set -- $parsed"
model='' output='' device=0 reference='' atol='' rtol='' env_file='' python=python3
warmup=0 repeat=1
inputs=() compare_args=()
while true; do
    case "$1" in
        -m|--model) model=$2; shift 2 ;;
        -i|--input) inputs+=(--input "$2"); shift 2 ;;
        -o|--output-dir) output=$2; shift 2 ;;
        -d|--device) device=$2; shift 2 ;;
        -r|--reference) reference=$2; shift 2 ;;
        -a|--atol) atol=$2; shift 2 ;;
        -t|--rtol) rtol=$2; shift 2 ;;
        -n|--output-map) compare_args+=(--output-map "$2"); shift 2 ;;
        -w|--warmup) warmup=$2; shift 2 ;;
        -b|--repeat) repeat=$2; shift 2 ;;
        -p|--python) python=$2; shift 2 ;;
        -e|--cann-env) env_file=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
    esac
done
[[ $# == 0 && -f $model && -n $output && ${#inputs[@]} -gt 0 ]] || { usage >&2; exit 2; }
if [[ -n $reference ]]; then
    [[ -f $reference/outputs.json ]] || fail "Reference manifest not found: $reference/outputs.json"
elif [[ -n $atol || -n $rtol || ${#compare_args[@]} -gt 0 ]]; then
    fail 'Comparison options require --reference.'
fi
load_cann "$env_file"
python_dir="$(dirname -- "${BASH_SOURCE[0]}")/../python"
"$python" "$python_dir/run_om.py" --model "$model" --device "$device" --output-dir "$output" \
    --warmup "$warmup" --repeat "$repeat" "${inputs[@]}"
if [[ -n $reference ]]; then
    "$python" "$python_dir/compare_outputs.py" --reference "$reference" --actual "$output" \
        --atol "${atol:-0.001}" --rtol "${rtol:-0.01}" "${compare_args[@]}"
fi
