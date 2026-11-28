#!/usr/bin/env bash
source /opt/atc-tools/scripts/_shell.sh
toolkit=/usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f $toolkit ]]; then
    load_cann "$toolkit"
else
    load_cann ''
fi
set +u
for setup in "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/share/info/ascendnpu-ir/bin/set_env.sh" \
    /usr/local/Ascend/nnal/atb/set_env.sh; do
    if [[ -f $setup ]]; then source "$setup"; fi
done
set -u
export PYTHONPATH="/opt/atc-tools/src${PYTHONPATH:+:$PYTHONPATH}"
# Keep CANN's Python packages out of the ONNX Runtime process. Child vendor tools
# still inherit the configured CANN environment and import the project by path.
exec /opt/onnx/bin/python -I -c 'import runpy, sys; sys.path.insert(0, "/opt/atc-tools/src"); runpy.run_module("atc_tools.executor", run_name="__main__")'
