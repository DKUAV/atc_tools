#!/usr/bin/env bash
set -euo pipefail

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

usage() {
    printf '%s\n' \
        'Usage: bash deploy/pack-offline.sh -o NEW_DIR -i IMAGE_TAG' \
        '       bash deploy/pack-offline.sh NEW_DIR IMAGE_TAG' \
        '  -o, --output-dir  New directory for the offline bundle' \
        '  -i, --image       Tested Linux ARM64 application image tag' \
        '  -h, --help        Show this help'
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

parsed=$(parse_options 'o:i:h' 'output-dir:,image:,help' "$@") || exit 2
eval "set -- $parsed"
output='' image=''
while true; do
    case "$1" in
        -o|--output-dir) output=$2; shift 2 ;;
        -i|--image) image=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
    esac
done

if [[ -z $output && -z $image && $# == 2 ]]; then
    output=$1
    image=$2
    shift 2
fi

if [[ $# != 0 || -z $output || -z $image ]]; then
    usage >&2
    exit 2
fi
if [[ -e $output ]]; then
    fail "Output directory already exists: $output"
fi

deployment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")
[[ $platform == linux/arm64 ]] || fail 'Expected a native ARM64 image'
mkdir -- "$output"
cp -- "$deployment_dir/compose.yaml" "$output/compose.yaml"
cp -- "$deployment_dir/lab.env.example" "$output/lab.env.example"
cp -- "$deployment_dir/README.md" "$output/README.md"
cp -- "$deployment_dir/README_zh.md" "$output/README_zh.md"
printf '%s\t%s\n' "$image" "$(docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$image")" > "$output/images.txt"
docker image save --output "$output/images.tar" "$image"
(
    cd -- "$output"
    sha256sum compose.yaml lab.env.example README.md README_zh.md images.txt images.tar > SHA256SUMS
)
