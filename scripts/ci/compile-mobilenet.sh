#!/usr/bin/env bash
set -euo pipefail
model="$RUNNER_TEMP/mobilenetv2-12.onnx"
revision=4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177
source="https://media.githubusercontent.com/media/onnx/models/$revision/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
curl --fail --silent --show-error --location --retry 3 --max-time 120 --output "$model" "$source"
echo "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5  $model" | sha256sum -c -
bash scripts/ci/service-smoke.sh compile "$model" "$RUNNER_TEMP/mobilenet-logs"
