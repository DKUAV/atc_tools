#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ${1:-} == --help ]]; then
    printf '%s\n' 'Usage: bash deploy/pack-offline.sh NEW_DIRECTORY [310B4_EXECUTOR [910B2_EXECUTOR]]' \
        'Build or pull the images for the destination CPU architecture before packing.'
    if [[ $# == 0 ]]; then exit 2; fi
    exit 0
fi
if [[ $# -gt 3 || -e $1 ]]; then
    printf '%s\n' 'Provide a new output directory and at most two executor images.' >&2
    exit 2
fi

deployment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=$1
inspection=${ATC_INSPECT_IMAGE:-atc-executor:local}
postgres=atc-tools-postgres:17
postgres_source=${ATC_POSTGRES_IMAGE:-$(sed -n 's/^    image: ${ATC_POSTGRES_IMAGE:-\(.*\)}$/\1/p' "$deployment_dir/compose.yaml")}
images=(atc-service:local "$inspection" "$postgres")
if [[ -n ${2:-} ]]; then images+=("$2"); fi
if [[ -n ${3:-} ]]; then images+=("$3"); fi

docker image inspect "$postgres_source" "${images[0]}" "$inspection" "${images[@]:3}" >/dev/null
platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "${images[0]}")
for ref in "$postgres_source" "$inspection" "${images[@]:3}"; do
    actual=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$ref")
    if [[ $actual != "$platform" ]]; then
        printf 'Image %s uses %s. All bundle images must use %s.\n' "$ref" "$actual" "$platform" >&2
        exit 2
    fi
done
docker image tag "$postgres_source" "$postgres"
mkdir -- "$output"
cp -- "$deployment_dir/compose.yaml" "$output/compose.yaml"
cp -- "$deployment_dir/lab.env.example" "$output/lab.env.example"
cp -- "$deployment_dir/OFFLINE.md" "$output/OFFLINE.md"
for ref in "${images[@]}"; do
    printf '%s\t%s\n' "$ref" "$(docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}}' "$ref")"
done > "$output/images.txt"
docker image save --output "$output/images.tar" "${images[@]}"
(
    cd -- "$output"
    sha256sum compose.yaml lab.env.example OFFLINE.md images.txt images.tar > SHA256SUMS
)
