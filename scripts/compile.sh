#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_shell.sh"

usage() {
    printf '%s\n' 'Usage: bash scripts/compile.sh -m ONNX -s SOC -o NEW_DIR [options]' \
        '  -m, --model                       ONNX model' \
        '  -s, --soc                         Exact ATC target SoC' \
        '  -o, --output-dir                  New directory for model.om and logs' \
        '  -i, --input-shape                 ATC input shape override' \
        '  -e, --cann-env                    CANN set_env.sh' \
        '  -c, --compression-optimize-conf   Native ATC compression config' \
        '  -a, --atc-arg                     One extra ATC argument, repeatable' \
        '  -h, --help                        Show this help'
}

parsed=$(parse_options 'm:s:o:i:e:c:a:h' 'model:,soc:,output-dir:,input-shape:,cann-env:,compression-optimize-conf:,atc-arg:,help' "$@") || exit 2
eval "set -- $parsed"
model='' soc='' output='' shape='' env_file='' compression=''
extra=()
while true; do
    case "$1" in
        -m|--model) model=$2; shift 2 ;;
        -s|--soc) soc=$2; shift 2 ;;
        -o|--output-dir) output=$2; shift 2 ;;
        -i|--input-shape) shape=$2; shift 2 ;;
        -e|--cann-env) env_file=$2; shift 2 ;;
        -c|--compression-optimize-conf) compression=$2; shift 2 ;;
        -a|--atc-arg) extra+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
    esac
done
[[ $# == 0 && -f $model && -n $soc && -n $output ]] || { usage >&2; exit 2; }
load_cann "$env_file"
command -v atc >/dev/null || fail 'atc is unavailable in the selected environment.'
args=(atc "--model=$model" --framework=5 "--soc_version=$soc" "--output=$output/model")
if [[ -n $shape ]]; then args+=("--input_shape=$shape"); fi
if [[ -n $compression ]]; then
    [[ -f $compression ]] || fail "Compression config not found: $compression"
    args+=("--compression_optimize_conf=$compression")
fi
for arg in "${extra[@]}"; do
    case "$arg" in
        --model|--model=*|--framework|--framework=*|--soc_version|--soc_version=*|--output|--output=*|--input_shape|--input_shape=*|--compression_optimize_conf|--compression_optimize_conf=*)
            fail "Use the dedicated option for $arg" ;;
        --*=*) args+=("$arg") ;;
        *) fail "Use --option=value for extra ATC arguments: $arg" ;;
    esac
done
mkdir -- "$output" || fail "Choose a new output directory: $output"
if [[ -n $compression ]]; then cp -- "$compression" "$output/compression-optimize.cfg"; fi
run_logged "$output" "${args[@]}"
[[ -s $output/model.om ]] || fail "ATC returned success without a nonempty $output/model.om."
