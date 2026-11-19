#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_amct.sh"
parse_amct_version "$@"

cann_repo=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN
ort_repo=https://cdn.jsdelivr.net/gh/microsoft/onnxruntime
mkdir -p -- "$package_dir"
staging=$(mktemp -d -- "$package_dir/.download.XXXXXXXX")
trap 'rm -rf -- "$staging"' EXIT

for arch in aarch64 x86_64; do
    archive=$(amct_archive_name "$arch")
    url="$cann_repo/CANN%20$cann_version/$archive"
    curl -fLsS --retry 2 --connect-timeout 15 -o "$staging/$archive" "$url" ||
        fail "Could not download $archive. Check the CANN version and the official download page."
    members=$(tar -tzf "$staging/$archive") || fail "Invalid AMCT archive: $archive"
    if ! grep -Fxq "./amct_acl/cann-amct_acl_${cann_version}_linux-$arch.run" <<< "$members" ||
       ! grep -Fxq './amct_onnx/amct_onnx_op.tar.gz' <<< "$members" ||
       ! grep -Exq "\./amct_onnx/amct_onnx-[^/]+-py3-none-linux_$arch\.whl" <<< "$members"; then
        fail "$archive does not contain the expected AMCT ACL and ONNX packages."
    fi
done

for ort_version in 1.16.0 1.20.0; do
    mkdir -p -- "$staging/ort/$ort_version" "$package_dir/ort/$ort_version"
    for header in "${ort_headers[@]}"; do
        curl -fLsS --retry 2 --connect-timeout 15 -o "$staging/ort/$ort_version/$header" \
            "$ort_repo@v$ort_version/include/onnxruntime/core/session/$header"
        [[ -s $staging/ort/$ort_version/$header ]] || fail "Empty ONNX Runtime header: $header"
    done
done

for ort_version in 1.16.0 1.20.0; do
    mv -f -- "$staging/ort/$ort_version/"*.h "$package_dir/ort/$ort_version/"
done
mv -f -- "$staging/"*.tar.gz "$package_dir/"
printf 'AMCT packages and ONNX Runtime headers downloaded to %s\n' "$package_dir"
