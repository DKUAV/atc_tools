#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 || -e $1 ]]; then
    printf 'Usage: %s NEW_DIRECTORY TESTED_IMAGE_TAG\n' "$0" >&2
    exit 2
fi
deployment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=$1
image=$2
platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")
[[ $platform == linux/arm64 ]] || { echo "Expected a native ARM64 image" >&2; exit 1; }
mkdir -- "$output"
cp -- "$deployment_dir/compose.yaml" "$output/compose.yaml"
cp -- "$deployment_dir/lab.env.example" "$output/lab.env.example"
cp -- "$deployment_dir/OFFLINE.md" "$output/OFFLINE.md"
printf '%s\t%s\n' "$image" "$(docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$image")" > "$output/images.txt"
docker image save --output "$output/images.tar" "$image"
(
    cd -- "$output"
    sha256sum compose.yaml lab.env.example OFFLINE.md images.txt images.tar > SHA256SUMS
)
