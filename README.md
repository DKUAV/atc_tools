# 昇腾 ATC 离线部署工具

将 ONNX 模型编译为昇腾 OM，支持 INT8 校准、设备推理和与 CPU 参考输出的对比。

以下先在 910B 上验证浮点模型，再根据需要量化并部署到 310。完整的离线准备、量化、替代推理方式和排障见[部署详细指南](docs/deployment-guide.md)。

## 1. 准备

### 1.1 环境准备

#### 确认环境

确认目标芯片的**完整 SoC 型号**、板端 CANN 版本，以及匹配目标的编译工具链和算子包。如果不确定版本，可以执行下面的环境检查脚本：

```bash
bash scripts/probe_env.sh
```

#### 安装依赖包

在仓库根目录激活 Python 3.10 或更高版本的虚拟环境后，以 editable 模式安装：

```bash
python3 -m pip install -e '.[prepare,yolo]'
atc_tools --help
```

`prepare` 提供 ONNX 模型检查和 CPU 参考推理依赖，`yolo` 提供直接输入图片以及 YOLO 示例所需的 OpenCV。安装时根据 Git 标签和提交生成包版本。 [依赖配置](pyproject.toml)

Python 命令统一使用 `atc_tools`。`onnx` 用于检查模型、生成 CPU 参考输出和准备校准数据，`om` 用于设备推理和输出对比，`yolo` 用于单图检测。

#### 安装 AMCT

量化需要 AMCT，CANN 默认不带 AMCT，如果缺少 AMCT，可以使用

```bash
bash scripts/download_amct.sh <CANN_version>
```

下载完成后，将 `work/amct/<CANN_version>/` 目录复制到目标机器的仓库相同路径。
在目标机器上，激活已准备好依赖的 Python 环境后，执行安装：

```bash
bash scripts/install_amct.sh <CANN_version>
```

### 1.2 数据准备

- 准备 ONNX 模型及外部权重、验证输入（`.npy`、`.bin`、`.jpg` 或 `.png`），以及生成 CPU 参考输出所需的环境。
- 离线使用时，提前准备镜像、模型、Python 依赖及需要的 CANN 软件包。
- INT8 还需要有代表性的校准数据集 (Calibration set) 。

## 2. 检查模型

### 2.2 模型检查与生成 CPU 参考输出

```bash
# Inspect model inputs, outputs, and external weights.
atc_tools onnx inspect --model work/model.onnx

# Generate CPU reference outputs for comparison.
atc_tools onnx reference --model work/model.onnx \
  --input images=work/input.npy --output-dir work/reference
```

这里的 `images` 是示例输入名，请按检查结果替换。多输入模型重复传入 `--input '输入名=文件路径'`。CPU 参考和 OM 推理可用同一个文件，如 `--input images=work/test.jpg`。JPG 和 PNG 会自动按 YOLO 规则预处理，其他预处理流程可生成与模型 shape 和 dtype 匹配的 NPY 或 BIN。 [输入格式和图片预处理](docs/deployment-guide.md#准备模型输入和-cpu-参考输出)

## 3. 本机编译并验证

### 3.1 编译浮点模型

#### 编译模型

将示例型号替换为实际完整型号，使用新的输出目录：

```bash
SOC_VERSION=Ascend910B2
bash scripts/compile.sh --model work/model.onnx --soc "${SOC_VERSION}" \
  --output-dir "work/${SOC_VERSION}-float"

```

要显式选择 FP16 计算模式，可通过 `--atc-arg` 传入 `--precision_mode_v2=fp16`，并使用独立的输出目录。 [ATC 精度模式](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0069.html)

```bash
bash scripts/compile.sh --model work/model.onnx --soc "${SOC_VERSION}" \
  --atc-arg=--precision_mode_v2=fp16 \
  --output-dir "work/${SOC_VERSION}-fp16"
```

OM 的输入和输出类型可另行指定。

- 输入设为 FP16 时，追加 `--atc-arg=--input_fp16_nodes=images`。把 `images` 换成 ONNX 的实际输入名，并为推理准备 `float16` 的 `.npy` 或 `.bin`。 [ATC 输入类型](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/atctool/atlasatcparam_16_0043.html)
- 输出设为 FP16 时，追加 `--atc-arg=--output_type=FP16`。 [ATC 输出类型](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0030.html)

#### 校验模型结果

可以通过下面命令检查编译后的模型在目标设备上的推理结果，并与 CPU 参考输出进行对比。

```bash
# Run the OM and compare its outputs with the CPU reference.
bash scripts/run_model.sh --model work/"$SOC_VERSION"-float/model.om --device 0 \
  --input images=work/input.npy --output-dir work/"$SOC_VERSION"-float-outputs \
  --reference work/reference
```

校验 FP16 版本时，将 `--model` 指向 `work/${SOC_VERSION}-fp16/model.om`，并为 `--output-dir` 选择新的目录。

编译目录保存 OM、命令、日志和退出码；推理目录保存原始输出，对比结果显示在终端。默认浮点容差仅用于排查，验收阈值应依据模型和业务确定。输入或输出名称变化、动态形状及容差设置见[详细指南](docs/deployment-guide.md#910b-上编译并执行浮点基线)。

### 3.2 编译量化模型

下面从 `work/model.onnx` 生成 INT8 OM。两种方式使用同一份校准数据，按需要选择一种。

#### 准备校准数据

选取能代表实际业务输入的样本，按模型要求完成预处理。每个 batch 放一个目录，按 ONNX 输入顺序命名为 `input_0.npy`、`input_1.npy` 等。每个张量的 shape 和 dtype 应与原始 ONNX 输入一致。以下用两个 batch 演示，正式量化按需要增加样本，并单独留出验证集。

```bash
atc_tools onnx calibration --model work/model.onnx \
  --sample work/samples/0001 --sample work/samples/0002 \
  --output-dir work/calibration
```

命令会检查输入张量，并生成逐 batch 的 BIN、按输入拼接的 BIN，以及记录参数的 `work/calibration/calibration.json`。 [校准数据格式](docs/deployment-guide.md#amct-onnx-预校准)

#### 使用 AMCT-ACL

ATC 在编译时调用 AMCT-ACL 完成校准并生成 OM。使用支持内联校准的 CANN 环境，确认已安装配套的 AMCT-ACL，且校准设备对当前进程可见。 [ATC 压缩配置](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0059.html)

```bash
cp configs/atc-compression.cfg work/atc-compression.cfg
```

编辑 `work/atc-compression.cfg` 的 `calibration` 配置：

| 字段 | 填写方式 |
| --- | --- |
| `input_data_dir` | 使用 `calibration.json` 的 `input_data_dir`，例如 `work/calibration/input_0.bin`。 |
| `input_shape` | 使用 `calibration.json` 的 `atc_input_shape`，其中 batch 维包含拼接后的样本。 |
| `infer_soc`、`infer_device_id` | 填校准设备的完整 SoC 型号和逻辑 ID，例如 `Ascend910B2` 和 `0`。 |

```bash
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_VERSION" \
  --compression-optimize-conf work/atc-compression.cfg \
  --output-dir "work/${SOC_VERSION}-int8-acl"
```

#### 使用 AMCT-ONNX

先激活与 AMCT-ONNX 配套的 Python 环境，确认 `amct_onnx calibration --help` 可用。复制量化配置，把 `batch_num` 设为实际使用的校准 batch 数，本例为 `2`。 [AMCT ONNX 环境](docs/deployment-guide.md#下载和安装-amct)

```bash
cp configs/amct-onnx.cfg work/amct-onnx.cfg
```

按 `calibration.json` 的 `input_shape`、`data_types` 和 `data_dir` 填写以下参数。示例使用一个名为 `images` 的 FP32 输入，shape 为 `[1,3,640,640]`。

```bash
bash scripts/quantize_model.sh --model work/model.onnx \
  --input-shape 'images:1,3,640,640' --data-types float32 \
  --data-dir work/calibration/input_0 --config work/amct-onnx.cfg \
  --output-dir work/quantized
```

量化生成的 `model_fake_quant_model.onnx` 可用于精度仿真。回到编译环境，用 `model_deploy_model.onnx` 生成 OM。

```bash
bash scripts/compile.sh --model work/quantized/model_deploy_model.onnx \
  --soc "$SOC_VERSION" --output-dir "work/${SOC_VERSION}-int8-onnx"
```

#### 验证量化结果

在同一设备上，用第 3.1 节的验证输入运行量化版 OM。以下以 AMCT-ONNX 产物为例，使用 AMCT-ACL 时将模型路径换成对应的编译目录。

```bash
bash scripts/run_model.sh --model "work/${SOC_VERSION}-int8-onnx/model.om" \
  --device 0 --input images=work/input.npy \
  --output-dir "work/${SOC_VERSION}-int8-outputs" \
  --reference "work/${SOC_VERSION}-float-outputs"
```

张量误差用于排查量化差异。精度验收应在同一份独立验证集上对比两份 OM 的任务指标，例如检测模型的 mAP，并使用相同的前处理和后处理。 [量化验收说明](docs/deployment-guide.md#编译-310-并在目标机验收)

## 4. 交叉编译模型

不需要量化时可直接使用原始 ONNX。需要 INT8 时，先准备代表性校准数据和与 CANN 版本匹配的 AMCT 环境。例如，联网下载并在目标 Linux 环境离线安装 CANN 9.2.0 对应的 AMCT：

```bash
# 在联网机器下载，并将 work/amct/9.2.0/ 复制到目标机器的仓库相同路径
bash scripts/download_amct.sh 9.2.0

# 在目标机器激活已准备好依赖的 Python 环境后安装
source /path/to/amct-venv/bin/activate
bash scripts/install_amct.sh 9.2.0
```

版本和路径按实际环境替换。选用其他 Toolkit 版本或自定义安装目录时，先加载对应的 `set_env.sh`。安装还需 C++ 编译器、Python 开发头文件及 Toolkit 安装权限。下载内容、依赖配套和离线安装步骤见 [下载和安装 AMCT](docs/deployment-guide.md#下载和安装-amct)。然后生成量化部署 ONNX，在 910B 比较浮点和量化结果；校准步骤和配置见 [INT8 量化](docs/deployment-guide.md#int8-量化)。

使用匹配目标芯片和板端 CANN 的编译环境，替换为实际 SoC 型号：

```bash
# Compile the ONNX model for the exact target 310 model.
SOC_310=Ascend310P3
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_310" \
  --output-dir work/310-float
```

将 OM、相同输入和参考输出传到目标 310，按上面的推理方式实际运行并对比。若使用 INT8，也要编译量化部署 ONNX，并在目标机验证精度和性能。编译成功本身不代表目标机验收通过；详见 [310 验收](docs/deployment-guide.md#编译-310-并在目标机验收)。

## 5. 更多

- [YOLO 单图示例](docs/deployment-guide.md#yolo-单图示例)
- [ais_bench 推理](docs/deployment-guide.md#用-ais_bench-的-python-api-执行-om)
- [排障](docs/deployment-guide.md#排障)与[本地开发检查](docs/deployment-guide.md#本地开发检查)
