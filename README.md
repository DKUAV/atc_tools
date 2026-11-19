# 昇腾 ATC 离线部署工具

将 ONNX 模型编译为昇腾 OM，支持 INT8 校准、设备推理和与 CPU 参考输出的对比。

以下先在 910B 上验证浮点模型，再根据需要量化并部署到 310。完整的离线准备、量化、替代推理方式和排障见[部署详细指南](docs/deployment-guide.md)。

## 准备

- 准备 ONNX 模型及外部权重、验证输入（`.npy`、`.bin`、`.jpg` 或 `.png`），以及生成 CPU 参考输出所需的环境。
- 确认 910B 和目标 310 的**完整 SoC 型号**、板端 CANN 版本，以及匹配目标的编译工具链和算子包。
- 离线使用时，提前准备镜像、模型、Python 依赖及需要的 CANN 软件包。
- INT8 还需要有代表性的校准样本和 AMCT 环境。具体清单和 wheel 准备方法见[部署详细指南](docs/deployment-guide.md#进机房前)。
- Shell 脚本依赖 Bash 和增强版 util-linux `getopt`；在 Bash 或 Zsh 中均使用 `bash scripts/脚本名.sh`。各脚本支持 `--help`。

## 1. 检查环境和模型

从仓库根目录执行。按实际情况加载 CANN 环境；编译机需要 ATC 和对应算子包，推理机还需要可见的 NPU 和配套的 pyACL。

```bash
# Check available tools, Python modules, and devices.
bash scripts/probe_env.sh

# Inspect model inputs, outputs, and external weights.
python3 python/prepare_model.py inspect --model work/model.onnx
# Generate CPU reference outputs for comparison.
python3 python/prepare_model.py reference --model work/model.onnx \
  --input images=work/input.npy --output-dir work/reference
```

这里的 `images` 是示例输入名，请按检查结果替换。多输入模型重复传入 `--input '输入名=文件路径'`。CPU 参考和 OM 推理可用同一个文件，如 `--input images=work/test.jpg`。JPG 和 PNG 会自动按 YOLO 规则预处理，其他预处理流程可生成与模型 shape 和 dtype 匹配的 NPY 或 BIN。必要时给 Shell 脚本传入 `--cann-env /实际安装目录/set_env.sh`。 [输入格式和图片预处理](docs/deployment-guide.md#准备模型输入和-cpu-参考输出)

## 2. 在 910B 编译并验证

将示例型号替换为实际完整型号，使用新的输出目录：

```bash
# Set the exact 910B model and compile the ONNX model.
SOC_910B=Ascend910B1
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_910B" \
  --output-dir work/910b-float
# Run the OM and compare its outputs with the CPU reference.
bash scripts/run_model.sh --model work/910b-float/model.om --device 0 \
  --input images=work/input.npy --output-dir work/910b-float-outputs \
  --reference work/reference
```

编译目录保存 OM、命令、日志和退出码；推理目录保存原始输出，对比结果显示在终端。默认浮点容差仅用于排查，验收阈值应依据模型和业务确定。输入或输出名称变化、动态形状及容差设置见[详细指南](docs/deployment-guide.md#910b-上编译并执行浮点基线)。

## 3. 按需量化并部署到 310

不需要量化时可直接使用原始 ONNX。需要 INT8 时，先准备代表性校准数据和与 CANN 版本匹配的 AMCT 环境。例如，联网下载并在目标 Linux 环境离线安装 CANN 9.2.0 对应的 AMCT：

```bash
# 在联网机器下载，并将 work/amct/9.2.0/ 复制到目标机器的仓库相同路径
bash scripts/download_amct.sh 9.2.0

# 在目标机器加载对应 CANN 和已准备好依赖的 Python 环境后安装
source /usr/local/Ascend/cann/set_env.sh
source /path/to/amct-venv/bin/activate
bash scripts/install_amct.sh 9.2.0
```

版本和路径按实际环境替换；安装还需 C++ 编译器、Python 开发头文件及 Toolkit 安装权限。下载内容、依赖配套和离线安装步骤见 [下载和安装 AMCT](docs/deployment-guide.md#下载和安装-amct)。然后生成量化部署 ONNX，在 910B 比较浮点和量化结果；校准步骤和配置见 [INT8 量化](docs/deployment-guide.md#int8-量化)。

使用匹配目标芯片和板端 CANN 的编译环境，替换为实际 SoC 型号：

```bash
# Compile the ONNX model for the exact target 310 model.
SOC_310=Ascend310P3
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_310" \
  --output-dir work/310-float
```

将 OM、相同输入和参考输出传到目标 310，按上面的推理方式实际运行并对比。若使用 INT8，也要编译量化部署 ONNX，并在目标机验证精度和性能。编译成功本身不代表目标机验收通过；详见 [310 验收](docs/deployment-guide.md#编译-310-并在目标机验收)。

## 更多

- [YOLO 单图示例](docs/deployment-guide.md#yolo-单图示例)
- [ais_bench 推理](docs/deployment-guide.md#用-ais_bench-的-python-api-执行-om)
- [排障](docs/deployment-guide.md#排障)与[本地开发检查](docs/deployment-guide.md#本地开发检查)
