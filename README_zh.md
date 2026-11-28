# 昇腾 ATC 离线部署工具

[English](README.md) | [中文](README_zh.md)

将 ONNX 模型编译为昇腾 OM，支持 INT8 校准、设备推理和与 CPU 参考输出的对比。

以下先在 910B 上验证浮点模型，再根据需要量化并部署到 310。完整的离线准备、量化、替代推理方式和排障见[部署详细指南](docs/guides/deployment-guide_zh.md)。

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
python3 -m pip install -e .
atc_tools --help
```

默认安装包含 ONNX 模型检查、CPU 参考推理、直接输入图片和 YOLO 示例所需的 Python 依赖。安装时根据 Git 标签和提交生成包版本。 [依赖配置](pyproject.toml)

模型工具使用 `atc_tools` 命令。`onnx` 用于检查模型、生成 CPU 参考输出和准备校准数据，`om` 用于设备推理和输出对比，`yolo` 用于单图检测。

同一个 wheel 也提供 `atc_service` 命令。运行服务需要安装 `service` 可选依赖。在仓库根目录执行：

```bash
python3 -m pip install -e '.[service]'
atc_service --help
```

服务启动、Web 工作台和 API 的使用方法见[服务使用指南](docs/guides/service-guide_zh.md)。

#### 安装 AMCT

量化需要 AMCT，CANN 默认不带 AMCT，如果缺少 AMCT，可以使用

```bash
bash scripts/download_amct.sh <CANN_version>
```

下载完成后，将 `work/amct/<CANN_version>/` 目录复制到目标机器的仓库相同路径。
在目标机器上，激活已准备好配套依赖的独立 AMCT Python 环境后，执行安装。 [AMCT 依赖版本](docs/guides/deployment-guide_zh.md#下载和安装-amct)

```bash
bash scripts/install_amct.sh <CANN_version>
```

### 1.2 数据准备

- 准备 ONNX 模型及外部权重、验证输入（`.npy`、`.bin`、`.jpg` 或 `.png`），以及生成 CPU 参考输出所需的环境。
- 离线使用时，提前准备镜像、模型、Python 依赖及需要的 CANN 软件包。
- INT8 还需要有代表性的校准数据集 (Calibration set) 。

## 2. 检查模型

### 2.1 模型检查与生成 CPU 参考输出

```bash
# Inspect model inputs, outputs, and external weights.
atc_tools onnx inspect --model work/model.onnx

# Generate CPU reference outputs for comparison.
atc_tools onnx reference --model work/model.onnx \
  --input images=work/input.npy --output-dir work/reference
```

这里的 `images` 是示例输入名，请按检查结果替换。多输入模型重复传入 `--input '输入名=文件路径'`。CPU 参考和 OM 推理可用同一个文件，如 `--input images=work/test.jpg`。JPG 和 PNG 会自动按 YOLO 规则预处理，其他预处理流程可生成与模型 shape 和 dtype 匹配的 NPY 或 BIN。 [输入格式和图片预处理](docs/guides/deployment-guide_zh.md#准备模型输入和-cpu-参考输出)

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

编译目录保存 OM、命令、日志和退出码；推理目录保存原始输出，对比结果显示在终端。默认浮点容差仅用于排查，验收阈值应依据模型和业务确定。输入或输出名称变化、动态形状及容差设置见[详细指南](docs/guides/deployment-guide_zh.md#910b-上编译并执行浮点基线)。

### 3.2 编译量化模型

下面从 `work/model.onnx` 生成 INT8 OM。单输入图片模型可直接走 [AMCT-ONNX 一键量化](#使用-amct-onnx)。使用 AMCT-ACL 或已有张量数据时，先准备下面的校准数据。

#### 准备校准数据

选取能代表实际业务输入的样本，按模型要求完成预处理。每个 batch 放一个目录，按 ONNX 输入顺序命名为 `input_0.npy`、`input_1.npy` 等。每个张量的 shape 和 dtype 应与原始 ONNX 输入一致。以下用两个 batch 演示，正式量化按需要增加样本，并单独留出验证集。

```bash
atc_tools onnx calibration --model work/model.onnx \
  --sample work/samples/0001 --sample work/samples/0002 \
  --output-dir work/calibration
```

命令会检查输入张量，并生成逐 batch 的 BIN、按输入拼接的 BIN，以及记录参数的 `work/calibration/calibration.json`。 [校准数据格式](docs/guides/deployment-guide_zh.md#amct-onnx-预校准)

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

先激活与 AMCT-ONNX 配套的 Python 环境，确认 `amct_onnx calibration --help` 可用。 [AMCT ONNX 环境](docs/guides/deployment-guide_zh.md#下载和安装-amct)

**从图片目录一键量化。** 把 JPG、JPEG 或 PNG 平铺在一个目录中，图片文件名可以自定。命令会从 ONNX 读取输入名、shape 和 dtype，按文件名顺序逐张预处理并调用 AMCT-ONNX。

```bash
/path/to/atc-tools-venv/bin/atc_tools onnx quantize --model work/model.onnx \
  --dataset work/images --output-dir work/quantized
```

将 `/path/to/atc-tools-venv` 换成安装了本工具的虚拟环境路径。`atc_tools` 会从当前 `PATH` 调用 AMCT 环境中的 `amct_onnx`。

适用于固定尺寸、batch 为 1 的单输入 RGB 模型，支持 NCHW 和 NHWC。预处理使用 YOLO 规则，等比缩放并填充 114，转换为 RGB，浮点输入除以 255。模型使用其他预处理时，先自行生成张量，再走下面的流程。

默认使用全部图片。加 `--batch-num 50` 可只用前 50 张。要设置跳过层、量化算法等参数，可加 `--config work/amct-onnx.cfg`，校准张数由原生配置中的 `batch_num` 指定。这两个参数二选一。省略 `--output-dir` 时，输出目录为模型旁边的 `<模型名>-int8/`。 [AMCT-ONNX 配置字段](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/devaids/amct/atlasamct_16_0168.html)

每次校准推理只处理一张图，模型的 batch size 保持为 1。首次量化可以用 `--batch-num 16` 试跑，观察内存用量和验证集精度后再调整张数。进程被杀时先查看系统 OOM 记录和容器或作业的内存上限。 [量化内存排查](docs/guides/deployment-guide_zh.md#amct-onnx-进程被杀)

**从已预处理的张量量化。** 复制量化配置，把 `batch_num` 设为实际使用的校准 batch 数，本例为 `2`。

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

两种流程都会生成 `model_deploy_model.onnx` 和 `model_fake_quant_model.onnx`。部署模型用于 ATC 编译，仿真模型用于精度仿真。一键量化的 AMCT 日志保存在输出目录中的 `amct-onnx.log`。

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

张量误差用于排查量化差异。精度验收应在同一份独立验证集上对比两份 OM 的任务指标，例如检测模型的 mAP，并使用相同的前处理和后处理。 [量化验收说明](docs/guides/deployment-guide_zh.md#编译-310-并在目标机验收)

## 4. 交叉编译模型

在 910B 上验证模型后，可以在 Linux 编译机上为目标 310 重新生成 OM。普通 ONNX 模型的 ATC 转换无需目标 NPU 参与。先确认板端的完整 SoC 型号和 CANN 版本，再选择与板端兼容的 Toolkit。CANN 8.5.0 及之后版本还需要安装与目标芯片匹配的 `ops` 算子包。 [ATC `--soc_version` 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0036.html)

**在编译机上生成 OM。** 以下以 `Ascend310B4` 为例，请换成板端的实际型号。`--soc` 对应 ATC 的 `--soc_version`。如果编译机装了多个 CANN 版本，可以用 `--cann-env` 指定对应的 `set_env.sh`。

```bash
TARGET_SOC=Ascend310B4
bash scripts/compile.sh --model work/model.onnx --soc "$TARGET_SOC" \
  --output-dir "work/${TARGET_SOC}-float"
```

需要 INT8 时，可用第 3.2 节 AMCT-ONNX 生成的 `work/quantized/model_deploy_model.onnx` 作为输入，保持目标 `--soc`，并使用独立的输出目录。如果量化 ONNX 与目标编译环境不兼容，用兼容的 AMCT 环境重新量化。 [310 部署指南](docs/guides/deployment-guide_zh.md#编译-310-并在目标机验收)

**在目标机上验收。** 将 `model.om`、验证输入和 CPU 参考输出复制到目标 310，按第 3.1 节运行并对比输出。部署 INT8 时，再按第 3.2 节用同一验证集比较目标机上的浮点版和 INT8 版，记录精度和性能。 [推理与输出对比](#校验模型结果) · [量化验收](#验证量化结果)

## 5. 服务部署与使用

除命令行 `atc_tools` 外，本仓库还提供容器化的 `atc_service` 服务，通过浏览器工作台（Web Workbench）和 HTTP API 完成 ONNX 检查、OM 编译、INT8 量化、推理、精度对比与 `msprof` 性能分析。完整配置、真实昇腾接入与冒烟测试见[服务使用指南](docs/guides/service-guide_zh.md)，离线包导入与升级见[离线部署指南](deploy/README_zh.md)。

### 5.1 部署与启停服务

复制 [deploy/lab.env.example](deploy/lab.env.example) 为 `lab.env`，按环境填写 `ATC_APP_IMAGE`、`ATC_HOST_DATA_DIR` 和 `SSH_CONFIG_DIR`（包含非空 `authorized_keys` 的 root 专用目录）。

- **本地模拟体验**：在 `lab.env` 中设置 `ATC_SIMULATE_COMPILE=1`（无需昇腾硬件），本机 HTTP 浏览器测试设置 `ATC_BROWSER_SECURE_COOKIES=0`。
- **真实昇腾环境**：保持 `ATC_SIMULATE_COMPILE=0`，按已验证目标配置 `ATC_AMCT_TARGETS`；启用 910B2 设备推理与性能分析需设置 `ATC_RUNTIME=ascend` 和 `ATC_910B2_DEVICE`。 [接入真实昇腾工具链](docs/guides/service-guide_zh.md#接入真实昇腾工具链)

```bash
# 启动服务（--wait 会等待容器通过 /readyz 健康检查）
docker compose --env-file lab.env -f deploy/compose.yaml up -d --no-build --pull never --wait

# 停止服务（保留数据目录）
docker compose --env-file lab.env -f deploy/compose.yaml down
```

### 5.2 使用 Web 工作台

先为浏览器账号设置或重置密码（仅使用 Web 工作台时无需签发 API Token）：

```bash
docker compose --env-file lab.env -f deploy/compose.yaml exec app atc_service set-browser-password alice
```

服务默认监听 `http://127.0.0.1:8766/`，使用上述账号密码登录。实验室或内网部署请通过 HTTPS 反向代理提供访问，并保持 `ATC_BROWSER_SECURE_COOKIES=1`。

1. 上传 ONNX 模型并等待自动检查完成。
2. 选择目标芯片（如 `310B4`、`910B2`）和精度预设（`original`、`mixed Float16`、`preferred FP16`、`INT8`）。
3. 按需上传数据集：选择 `INT8` 时上传校准集，选择 `910B2` 且开启精度对比或 `msprof` 性能分析时上传验证集（支持图片、`.npy`、`.bin`，单数据集最多 128 个样本）。
4. 提交任务后查看分阶段日志与状态，完成后下载 OM、`accuracy.json` 或 `profiling.zip` 等产物；可选验证阶段失败时可直接重试。

浏览器会话默认有效期为 12 小时，下载完所需产物后建议及时退出登录，以便及时清理会话文件。 [Web 工作台详情](docs/guides/service-guide_zh.md#使用浏览器工作台)

### 5.3 通过 HTTP API 提交任务

调用 HTTP API 需要先签发 Bearer Token（按需也可吊销）：

```bash
# 签发 API Bearer Token
docker compose --env-file lab.env -f deploy/compose.yaml exec -T app atc_service issue-token alice

# 吊销 Token（交互式选择）
docker compose --env-file lab.env -f deploy/compose.yaml exec app atc_service revoke-token
```

所有 API 请求均需携带 `Authorization: Bearer <token>` 请求头，传入稳定的 `Idempotency-Key` 可在中断时安全恢复任务，完整字段定义见 `GET /openapi.json`。上传模型后需先提交 `inspect_onnx` 检查任务，检查通过后再提交工作流或独立操作。

| 步骤 | 接口 | 说明 |
| --- | --- | --- |
| 查询能力 | `GET /v1/capabilities` | 获取可用预设、`workflow_presets` 和 `standalone_operations` |
| 上传模型 | `POST /v1/assets?filename=model.onnx` | 以 `application/octet-stream` 上传模型，获取 `input_id` |
| 上传数据集 | `POST /v1/dataset-files?filename=sample.npy` | 按需上传校准或验证样本文件，返回数据集资产 ID |
| 提交工作流 | `POST /v1/workflows` | 基于已检查的 `model_id` 提交完整转换、量化与验证流程 |
| 提交独立操作 | `POST /v1/operations` | 执行单步操作（`quantize_onnx`、`compile_om`、`run_cpu`、`run_npu`、`compare_outputs`、`profile_om`） |
| 轮询与日志 | `GET /v1/jobs/{job_id}` / `GET /v1/jobs/{job_id}/logs` | 查询任务状态与产物 ID，或传入 `next_offset` 增量读取日志 |
| 重试与下载 | `POST /v1/jobs/{job_id}/retry` / `GET /v1/assets/{asset_id}/download` | 重试 `compare` 或 `profile` 阶段，或下载任务产物 |

更详细的请求体示例和调用步骤见[通过 HTTP API 提交任务](docs/guides/service-guide_zh.md#通过-http-api-提交任务)。

## 6. 更多

- [服务使用指南](docs/guides/service-guide_zh.md)与[离线部署指南](deploy/README_zh.md)
- [YOLO 单图示例](docs/guides/deployment-guide_zh.md#yolo-单图示例)
- [ais_bench 推理](docs/guides/deployment-guide_zh.md#用-ais_bench-的-python-api-执行-om)
- [排障](docs/guides/deployment-guide_zh.md#排障)与[本地开发检查](docs/guides/deployment-guide_zh.md#本地开发检查)
