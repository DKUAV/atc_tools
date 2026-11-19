# 昇腾 ATC 离线部署工具

工具以 ONNX、OM 和 NumPy 张量为接口，支持检查模型、ATC 编译、INT8 训练后量化、pyACL 推理和输出对比。首版推理支持单设备、固定 shape 和连续存储的张量，可通过多个 `--input` 传入多输入模型。YOLO 是一套可替换的图像检测示例。

## 进机房前

- 准备 ONNX、外部权重文件、验证输入和 CPU 参考输出。需要 INT8 时，另备代表性校准样本、原生配置文件和量化部署 ONNX。
- 确认 910B 的完整 SoC 型号，以及目标 310 的完整型号和板端 CANN 版本。比如 `Ascend310`、`Ascend310P3`、`Ascend310B4` 是不同目标。编译需要与目标芯片匹配的工具链和算子包。
- 910B 可先用现有镜像 `2.10.0.post7-cann9.2.0-910b-ubuntu22.04-py3.12`。用 `docker image inspect` 确认镜像在离线服务器，按现场 NPU 容器运行方式挂载设备和工具目录。编译 310 所需的镜像或 CANN 软件包另行准备。
- Shell 入口用 Bash 和增强版 util-linux `getopt`。Bash 或 Zsh 终端都运行 `bash scripts/xxx.sh`。所有 Shell 入口支持短选项、长选项和 `--help`。由工具用 `--cann-env` 加载对应的 `set_env.sh`。
- 开发环境和 ONNX 准备工具使用 Python 3.10 或更高版本，YOLO 示例另需 OpenCV。仅运行 OM 和比对输出时，可以直接用板端的 Python 3.9，使用 NumPy 1.21 或更高版本和与板端 CANN、Python ABI 配套的厂商 pyACL。准备 wheel 时使用目标 Linux 架构和 Python 版本的联网环境。AMCT 使用其自身的依赖配套表。

在联网的开发环境可用 `uv sync --extra prepare --extra yolo` 安装 Python 工具依赖。下面命令可在与离线使用环境同架构、同 Python 版本的联网机器上准备 wheelhouse。

```bash
mkdir -p work
uv export --extra prepare --extra yolo --no-dev --no-emit-project \
  --format requirements-txt --output-file work/requirements-tools.txt
python3 -m pip download --only-binary=:all: -r work/requirements-tools.txt -d work/wheelhouse
```

离线环境中按实际用途安装。下面是仅执行 OM 时的示例。若板端 Python 低于 3.10，单独为板端下载适配的 NumPy wheel。

```bash
python3 -m pip install --no-index --find-links work/wheelhouse 'numpy>=1.21'
```

打包前可为交付目录生成校验清单，在目标机用 `sha256sum -c SHA256SUMS` 检查传输结果。例如在待传输目录中执行下列命令。

```bash
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
```

运行前确认实际使用的 Python。可用 `--python` 指向配套解释器。下面命令把探针结果保存到工作目录。

```bash
bash scripts/probe_env.sh |& tee work/environment.log
```

环境尚未加载时，给探针、编译和推理脚本传入 `--cann-env /实际安装目录/set_env.sh`。探针会逐项报告可用工具、环境变量、设备和 Python 模块。编译机主要检查 ATC 和算子包，推理机还需要设备可见和 `import acl` 成功。

## 准备模型输入和 CPU 参考输出

从仓库根目录执行命令。下文以单输入 `images` 为例，实际名称和形状以检查结果为准。

```bash
python3 python/prepare_model.py inspect --model work/model.onnx
python3 python/prepare_model.py reference --model work/model.onnx \
  --input images=work/input.npy --output-dir work/reference
```

检查命令显示 opset 及其算子域、输入输出、动态维度、外部权重文件和缺失的文件。生成参考输出时，每个输入传一次 `--input '输入名=文件路径.npy'`。张量应当已经完成模型自己的前处理，shape 和 dtype 要与 ONNX 一致。参考输出目录包含 `output_0.npy` 等文件和记录名称、shape、dtype 的 `outputs.json`。其它模型可直接复用这套接口。

## 910B 上编译并执行浮点基线

如果现场已配置 Ascend Docker Runtime，可以从工具目录进入现有镜像。镜像的仓库名按服务器上的本地标签填写，设备号按现场分配调整。

```bash
IMAGE=ghcr.io/dkuav/luciole-ascend:2.10.0.post7-cann9.2.0-910b-ubuntu22.04-py3.12
docker image inspect "$IMAGE" >/dev/null
docker run --pull=never --rm -it --runtime=ascend -e ASCEND_VISIBLE_DEVICES=0 \
  -v "$PWD:/work" -w /work --entrypoint /bin/bash "$IMAGE" -l
```

进入容器后先运行 `bash scripts/probe_env.sh`，再编译和推理。

```bash
SOC_910B=Ascend910B1  # Replace with the exact device SoC.
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_910B" \
  --output-dir work/910b-float
bash scripts/run_model.sh --model work/910b-float/model.om --device 0 \
  --input images=work/input.npy --output-dir work/910b-float-outputs \
  --reference work/reference
```

`compile.sh` 为 ONNX 使用 `--framework=5`。可传 `--input-shape 'images:1,3,640,640'`，额外 ATC 参数按 `--atc-arg=--参数名=参数值` 重复传入。编译和量化目录包含外部工具的原始退出码 `exit-code.txt`、日志 `tool.log` 和可供 Bash 重放的 `command.txt`。每次使用新的输出目录。路径中的空格由 Shell 入口保留，ATC、AMCT 以及它们的配置文件内部仍受厂商字符集限制。

`run_model.sh` 保存 OM 原始输出。提供 `--reference` 时按张量名对比，可用 `--output-map 'ONNX输出名=OM输出名'` 显式处理 ATC 重命名。若输入名变化，先看 runner 打印的名称差异，再用 OM 实际输入名传入同一个文件。

浮点使用 `abs(actual - ref) <= atol + rtol * abs(ref)`，默认 `--atol 0.001 --rtol 0.01` 只用于排查。浮点位宽允许不同，报告会同时显示两侧 dtype。整数和布尔值逐项精确比较。模型输出不匹配或超阈值时退出码非零，已保存的原始输出可独立重比。

### 用 ais_bench 的 Python API 执行 OM

安装昇腾的 OM 推理版 `ais_bench` 和配套 `aclruntime`。离线使用时提前把 wheel 和依赖放进 wheelhouse。按设备上的 CANN、CPU 架构和 Python ABI 选包，例如 ARM64 Python 3.12 环境对应 `cp312`、`linux_aarch64` 的 `aclruntime` wheel。执行前加载该环境的 CANN `set_env.sh`。 [官方安装与 wheel 清单](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/README.md)

```bash
python3 python/run_om_ais_bench.py --model work/910b-float/model.om --device 0 \
  --input images=work/input.npy --output-dir work/910b-ais-bench-outputs
python3 python/compare_outputs.py --reference work/reference \
  --actual work/910b-ais-bench-outputs
```

这个入口执行一次固定 shape 推理。多输入可重复传 `--input`，脚本按 OM 的输入顺序传给 `InferSession`。输出沿用 `output_0.npy` 和 `outputs.json` 格式，可以与 CPU 参考结果或 pyACL 的输出目录比较。需要预热、重复推理和吞吐统计时，可使用官方的 `python3 -m ais_bench` 命令。 [官方 Python API](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/API_GUIDE.md)

## INT8 量化

### AMCT ONNX 预校准

优先在联网环境装好并验证 `amct_onnx calibration --help`，以及一次实际校准。若需离线重做校准，带上已验证的 AMCT 环境。自定义算子库需要与 ONNX Runtime 版本配套，提前完成安装和测试。

校准数据应代表业务场景。每个校准 batch 建一个目录，按 ONNX 的输入顺序放 `input_0.npy`、`input_1.npy` 等。同一个输入在所有 batch 中保持 dtype、shape 和预处理一致。另留独立的验证集。

```bash
python3 python/prepare_model.py calibration --model work/model.onnx \
  --sample work/samples/0001 --sample work/samples/0002 --output-dir work/calibration
cp configs/amct-onnx.cfg work/amct-onnx.cfg
```

将 `work/amct-onnx.cfg` 中的 `batch_num` 按要参与校准的 batch 数填写。敏感层可在配置中增加 `skip_layers: "实际节点名"`。如需换校准算法或对单层覆盖策略，按所用 AMCT 的 `calibration_config_onnx.proto` 扩展该配置，保留最终使用的文件和量化日志。

```bash
bash scripts/quantize_model.sh --model work/model.onnx \
  --input-shape 'images:1,3,640,640' --data-types float32 \
  --data-dir work/calibration/input_0 --config work/amct-onnx.cfg \
  --output-dir work/quantized
bash scripts/compile.sh --model work/quantized/model_deploy_model.onnx \
  --soc "$SOC_910B" --output-dir work/910b-int8
```

`calibration.json` 中有对应的 `input_shape`、`data_types`、`data_dir`。多输入按清单中的顺序传入，Shell 中需引用含分号的整段值。`model_fake_quant_model.onnx` 用于 CPU 精度仿真，若依赖 AMCT 自定义算子库，给 `prepare_model.py reference` 加 `--custom-ops-library /path/to/library.so`，并使用配套的 ONNX Runtime。编译要选 `model_deploy_model.onnx`。

### ATC 内联校准

选择支持内联校准的 CANN 环境，确认 AMCT（ACL）已安装且校准 NPU 可见。复制 [ATC 原生模板](configs/atc-compression.cfg)，根据 `calibration.json` 填写 `input_data_dir` 和 `atc_input_shape`。这里的 shape 包括拼接后的校准 batch，`infer_soc` 和 `infer_device_id` 指校准设备。编译生成 OM 的目标仍由 `--soc` 控制。

```bash
cp configs/atc-compression.cfg work/atc-compression.cfg
# Fill in the calibration paths, shape, and device before compiling.
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_910B" \
  --compression-optimize-conf work/atc-compression.cfg --output-dir work/910b-int8-inline
```

若要传逐层策略，可在 `calibration` 内加 `config_file`，使用当前 CANN 的 AMCT（ACL）简易量化配置。内联校准和 AMCT ONNX 的原生配置各有参数格式。按版本文档核对校准样本组织方式及精度参数约束。 [ATC 压缩配置](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0059.html) · [AMCT ONNX 环境](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/devaids/amct/atlasamct_16_0014.html)

## 编译 310 并在目标机验收

进入已确认支持目标芯片和板端版本的编译环境。下面以 310P3 为示例，按设备完整型号替换。

```bash
SOC_310=Ascend310P3
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_310" \
  --output-dir work/310-float
bash scripts/compile.sh --model work/quantized/model_deploy_model.onnx \
  --soc "$SOC_310" --output-dir work/310-int8
```

将两份 OM、同一组输入和参考输出传到目标 310，分别按 910B 的命令运行并保存结果。编译产物和退出码记录编译是否成功，目标机推理日志与输出记录部署是否成功。若量化 ONNX 与目标 CANN 不兼容，回到目标版本支持的 AMCT 环境重新生成部署模型。

量化验收先对比同一设备的浮点和 INT8 输出。YOLO 用独立有标注的验证集、相同的预处理和后处理，使用项目的评测器记录两份 OM 的 mAP 及差值。用实测结果和业务允许的损失判定是否排除敏感层。性能对比时，分别为两份 OM 传入相同的 `--warmup 10 --repeat 50`，记录 `acl.mdl.execute` 的中位数。此计时范围只包含同步模型执行。

## YOLO 单图示例

```bash
python3 examples/yolo/yolo_image.py preprocess --image work/test.jpg \
  --width 640 --height 640 --output work/input.npy
# After running the OM, use outputs.json to locate the prediction tensor.
python3 examples/yolo/yolo_image.py postprocess --image work/test.jpg \
  --width 640 --height 640 --predictions work/910b-float-outputs/output_0.npy \
  --layout v8 --output-dir work/detections
```

预处理示例使用等比缩放和填充、RGB、NCHW、除以 255，默认输出 FP32，支持 `--dtype float16`。执行前核对模型原始预处理。后处理按实际导出布局选 `v5`（`[1,N,5+C]` 含 objectness）、`v8`（`[1,4+C,N]`）、或 `xyxy`（`[1,N,6]`，坐标、置信度、类别）。前两种执行按类别 NMS。产物为 `detections.json` 和 `annotated.jpg`。其他检测头、分割或姿态估计模型可沿用通用工具，再提供对应后处理。

## 排障

| 现象 | 检查 |
|---|---|
| ATC 找不到算子或目标型号 | 核对精确 `soc_version`、Toolkit、ops 包、ONNX opset 和失败日志。 |
| 有 ATC，pyACL 导入或设置设备失败 | 分别确认 Python ABI、CANN 运行库、逻辑设备号和容器设备挂载。 |
| 校准数据大小不符 | 核对输入顺序、dtype、每批 shape 和批数。ATC 内联校准用拼接后的 `atc_input_shape`。 |
| OM 描述的字节数、格式不符合输入 | 查看 ATC 的输入形状、格式和 AIPP 等编译选项。runner 需要能按描述还原的连续张量。 |
| 输出数值超限 | 先用完全相同的输入比较浮点 ONNX 与浮点 OM，再对比量化产物。可分别检查量化模拟和实机输出。 |

## 本地开发检查

```bash
uv run ty check
PYTHONPATH=python uv run python -m unittest discover -s tests -v
```

本地测试使用 ONNX Runtime CPU 和模拟 pyACL，可验证数据格式、参数传递所依赖的接口和资源释放。ATC、AMCT 与 NPU 推理按本文流程在目标环境验收。
