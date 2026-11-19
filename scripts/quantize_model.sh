#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_shell.sh"

usage() {
    printf '%s\n' 'Usage: bash scripts/quantize_model.sh -m ONNX -i SHAPES -d BIN_DIRS -t DTYPES -o NEW_DIR [options]' \
        '  -m, --model           Floating-point ONNX model' \
        '  -i, --input-shape     AMCT input names and dimensions' \
        '  -d, --data-dir        AMCT calibration .bin paths or directories' \
        '  -t, --data-types      AMCT input data types' \
        '  -o, --output-dir      New directory for quantized ONNX models and logs' \
        '  -c, --config          Native AMCT ONNX calibration config' \
        '  -e, --cann-env        CANN set_env.sh, if needed' \
        '  -h, --help            Show this help'
}

parsed=$(parse_options 'm:i:d:t:o:c:e:h' 'model:,input-shape:,data-dir:,data-types:,output-dir:,config:,cann-env:,help' "$@") || exit 2
eval "set -- $parsed"
model='' shape='' data='' types='' output='' config='' env_file=''
while true; do
    case "$1" in
        -m|--model) model=$2; shift 2 ;;
        -i|--input-shape) shape=$2; shift 2 ;;
        -d|--data-dir) data=$2; shift 2 ;;
        -t|--data-types) types=$2; shift 2 ;;
        -o|--output-dir) output=$2; shift 2 ;;
        -c|--config) config=$2; shift 2 ;;
        -e|--cann-env) env_file=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
    esac
done
[[ $# == 0 && -f $model && -n $shape && -n $data && -n $types && -n $output ]] || { usage >&2; exit 2; }
load_cann "$env_file"
command -v amct_onnx >/dev/null || fail 'amct_onnx is unavailable in the selected environment.'
args=(amct_onnx calibration --model "$model" --input_shape "$shape" --data_dir "$data" \
      --data_types "$types" --save_path "$output/model")
if [[ -n $config ]]; then
    [[ -f $config ]] || fail "Calibration config not found: $config"
    args+=(--calibration_config "$config")
fi
mkdir -- "$output" || fail "Choose a new output directory: $output"
if [[ -n $config ]]; then cp -- "$config" "$output/amct-onnx.cfg"; fi
run_logged "$output" "${args[@]}"
for artifact in model_deploy_model.onnx model_fake_quant_model.onnx; do
    [[ -s $output/$artifact ]] || fail "AMCT returned success without a nonempty $output/$artifact."
done
