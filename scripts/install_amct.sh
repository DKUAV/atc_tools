#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_amct.sh"
parse_amct_version "$@"

arch=$(uname -m)
[[ $(uname -s) == Linux && ( $arch == aarch64 || $arch == x86_64 ) ]] ||
    fail 'AMCT installation requires Linux on aarch64 or x86_64.'
archive="$package_dir/$(amct_archive_name "$arch")"
[[ -s $archive ]] || fail "AMCT archive missing: $archive. Run download_amct.sh first."

# The ACL installer takes the parent of the versioned Toolkit directory.
toolkit=${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}
[[ -n $toolkit && -d $toolkit ]] || fail 'Source set_env.sh for the target CANN Toolkit first.'
toolkit=$(readlink -f -- "$toolkit")
if [[ ${toolkit##*/} != "cann-$cann_version" ]]; then
    info="$toolkit/$arch-linux/ascend_toolkit_install.info"
    if [[ ${toolkit##*/} != cann || ! -r $info ]] ||
       ! grep -Fixq "version=$cann_version" "$info"; then
        fail "The loaded CANN Toolkit does not match $cann_version: $toolkit"
    fi
fi

command -v python3 >/dev/null || fail 'Activate the target Python environment first.'
python=$(command -v python3)
ort_version=$("$python" -c 'import numpy, onnx, onnxruntime, google.protobuf, setuptools; print(onnxruntime.__version__)') ||
    fail 'Install the matching AMCT Python dependencies first. See README.md.'
case "$ort_version" in
    1.16.0|1.20.0) ;;
    *) fail "ONNX Runtime $ort_version is unsupported by this installer. Use 1.16.0 or 1.20.0." ;;
esac
for header in "${ort_headers[@]}"; do
    [[ -s $package_dir/ort/$ort_version/$header ]] ||
        fail "ONNX Runtime $ort_version header missing: $header. Run download_amct.sh first."
done
command -v c++ >/dev/null || fail 'A C++ compiler is required to build AMCT ONNX custom operators.'

staging=$(mktemp -d)
trap 'rm -rf -- "$staging"' EXIT
tar -xzf "$archive" -C "$staging" ./amct_onnx ./amct_acl
shopt -s nullglob
wheels=("$staging"/amct_onnx/amct_onnx-*-py3-none-linux_"$arch".whl)
acl_installer="$staging/amct_acl/cann-amct_acl_${cann_version}_linux-$arch.run"
[[ ${#wheels[@]} == 1 && -s $acl_installer ]] || fail "Unexpected AMCT package contents: $archive"
tar -xzf "$staging/amct_onnx/amct_onnx_op.tar.gz" -C "$staging"
cp -- "$package_dir/ort/$ort_version/"*.h "$staging/amct_onnx_op/inc/"

"$python" -m pip install --no-index --no-deps --force-reinstall "${wheels[0]}"
(
    cd -- "$staging/amct_onnx_op"
    "$python" setup.py build
)
"$python" -c 'import amct_onnx; raise SystemExit(amct_onnx.AMCT_SO is None)'
"$python" -c 'from amct_onnx.cmd_line_tools.command import main; main()' calibration --help

bash "$acl_installer" --quiet --run --install-path="$(dirname -- "$toolkit")"
[[ -s $toolkit/lib64/libamctacl.so || -s $toolkit/$arch-linux/lib64/libamctacl.so ]] ||
    fail "The AMCT ACL library was not found in $toolkit."
printf 'AMCT ONNX and ACL installed for CANN %s (%s).\n' "$cann_version" "$arch"
