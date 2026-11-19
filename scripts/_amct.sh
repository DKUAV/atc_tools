# Shared arguments and package layout for AMCT entry points.
source "$(dirname -- "${BASH_SOURCE[0]}")/_shell.sh"

parse_amct_version() {
    local parsed usage="Usage: bash scripts/${0##*/} CANN_VERSION"
    parsed=$(parse_options 'h' 'help' "$@") || exit 2
    eval "set -- $parsed"
    if [[ $1 == -h || $1 == --help ]]; then
        printf '%s\n' "$usage"
        exit 0
    fi
    shift
    if [[ $# != 1 || ! $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][[:alnum:]]+)*$ ]]; then
        printf '%s\n' "$usage" >&2
        exit 2
    fi
    cann_version=$1
    package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/work/amct/$cann_version"
}

amct_archive_name() {
    printf 'Ascend-cann-amct_%s_linux-%s.tar.gz\n' "$cann_version" "$1"
}

ort_headers=(onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h
             onnxruntime_float16.h onnxruntime_session_options_config_keys.h)
