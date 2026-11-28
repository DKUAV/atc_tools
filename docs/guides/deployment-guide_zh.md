# 昇腾 ATC 离线部署详细指南

[English](deployment-guide.md) | [中文](deployment-guide_zh.md)

本指南涵盖离线环境准备、模型检查与 CPU 参考输出、910B 浮点基线验证、INT8 量化（AMCT-ONNX / AMCT-ACL）、310 交叉编译与验收、YOLO 示例及常见排障。

> 当前推理入口面向**单设备、固定 shape、连续内存存储**的张量；多输入模型可通过重复传入 `--input` 支持。YOLO 示例仅作为可替换的视觉检测参考。

## 目录

- [进机房前](#进机房前)
- [准备模型输入和 CPU 参考输出](#准备模型输入和-cpu-参考输出)
- [910B 上编译并执行浮点基线](#910b-上编译并执行浮点基线)
- [INT8 量化](#int8-量化)
- [编译 310 并在目标机验收](#编译-310-并在目标机验收)
- [YOLO 单图示例](#yolo-单图示例)
- [排障](#排障)
- [本地开发检查](#本地开发检查)

---

## 进机房前

### 准备清单

- **模型与数据**：准备好 ONNX 模型及其外部权重文件、验证输入样本以及 CPU 参考输出。如果涉及 INT8 量化，还需准备有代表性的校准数据集、原生量化配置文件，或提前导出的量化部署 ONNX。
- **硬件与 CANN 版本**：确认 910B 与目标 310 的**完整 SoC 型号**和板端 CANN 版本。例如 `Ascend310`、`Ascend310P3`、`Ascend310B4` 属于不同目标，必须使用与芯片型号匹配的 CANN 工具链和算子包（`ops`）。
- **容器与运行环境**：910B 阶段可先复用已有镜像 `2.10.0.post7-cann9.2.0-910b-ubuntu22.04-py3.12`（用 `docker image inspect` 确认离线服务器本地已存在，并按现场规范挂载 NPU 设备和工具目录）。面向 310 的编译镜像或 CANN 软件包需单独准备。
- **Shell 与 Python 要求**：
  - 所有脚本入口均使用 Bash 和增强版 util-linux `getopt`，无论当前终端是 Bash 还是 Zsh，统一通过 `bash scripts/xxx.sh` 调用。所有脚本均支持 `-h` / `--help`，支持 `--cann-env` 的脚本可用来显式指定 `set_env.sh`。
  - Python 工具要求 **Python 3.10+**；直接传入 JPG/PNG 或运行 YOLO 示例需要 OpenCV；OM 推理需要与板端 CANN 及 Python ABI 匹配的厂商 `pyACL` 或 `ais_bench`；AMCT 依赖需单独按配套表准备。

### 准备离线依赖（Wheelhouse）

联网开发环境可直接用 `uv sync` 以 editable 模式安装。若目标机无外网，在与目标机**相同 Linux 架构和 Python 版本**的联网机器上下载 wheelhouse（包含 editable 构建所需的 `hatchling` 和 `hatch-vcs`）：

```bash
mkdir -p work
uv export --no-dev --no-emit-project \
  --format requirements-txt --output-file work/requirements-tools.txt
python3 -m pip download --only-binary=:all: -r work/requirements-tools.txt -d work/wheelhouse
python3 -m pip download --only-binary=:all: hatchling hatch-vcs -d work/wheelhouse
```

将仓库目录与 `work/wheelhouse` 拷贝到离线目标机后，在仓库根目录安装：

```bash
python3 -m pip install --no-index --find-links work/wheelhouse -e .
atc_tools --version
```

> 包版本在安装时根据 Git 标签和提交自动生成。离线拷贝工作区时保留 `.git`（或使用预构建的源码包）即可保留版本信息；若缺失版本元数据，包版本会回退为 `0.0.0`。OM 推理后端（`pyACL` / `ais_bench`）需另行安装。

### 校验传输文件与环境探针

打包交付目录前，建议生成 SHA-256 校验清单，传到目标机后执行 `sha256sum -c SHA256SUMS` 确认文件完整：

```bash
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
```

开始编译或推理前，先运行环境探针并留存日志（可通过 `--python` 指定目标 Python 解释器）：

```bash
bash scripts/probe_env.sh |& tee work/environment.log
```

**CANN 环境加载优先级**（适用于探针、编译、量化和推理脚本）：
1. 优先加载 `--cann-env /实际安装目录/set_env.sh` 显式指定的文件；
2. 其次沿用当前 Shell 中已加载的 CANN 环境变量；
3. 若均未设置，则依次自动查找 `/usr/local/Ascend/cann/set_env.sh` 和旧版路径 `/usr/local/Ascend/ascend-toolkit/set_env.sh`。

探针会逐项输出可用工具、环境变量、NPU 设备及 Python 模块状态：编译机重点看 ATC 与算子包是否就绪，推理机还需确认设备可见且 `import acl` 正常。

---

## 准备模型输入和 CPU 参考输出

以下命令均在仓库根目录执行。示例中假设模型只有一个名为 `images` 的输入，实际名称与形状以检查结果为准：

```bash
atc_tools onnx inspect --model work/model.onnx
atc_tools onnx reference --model work/model.onnx \
  --input images=work/input.npy --output-dir work/reference
```

`onnx inspect` 会打印模型的 opset 与算子域、输入输出信息、动态维度以及外部权重文件的完整性。执行参考推理和 OM 推理时，每个输入传入一次 `--input '输入名=文件路径'`，多输入模型可以混用以下格式：

| 格式 | 输入要求 |
| --- | --- |
| `.npy` | 已完成预处理的 NumPy 张量，shape 和 dtype 必须与模型定义一致。 |
| `.bin` | 按 C 顺序连续排列的原始二进制张量，需与模型 dtype 及运行环境字节序一致，文件字节数必须严格等于元素个数 × 单元素字节数（例如用 `array.tofile("input.bin")` 生成）。 |
| `.jpg`、`.jpeg`、`.png` | 适用于 NCHW 或 NHWC 布局明确的固定 shape 单张 RGB 输入。工具会按 YOLO 规则等比缩放并用 `114` 居中填充，浮点输入除以 `255`，`uint8` 输入保留原始像素值（默认依赖已包含 `opencv-python-headless`）。 |

如果模型使用其他图像前处理逻辑，请先自行生成 `.npy` 或 `.bin` 再传入。参考输出目录会生成 `output_0.npy` 等张量文件，以及记录输出名称、shape 和 dtype 的 `outputs.json`。

---

## 910B 上编译并执行浮点基线

如果服务器已配置 Ascend Docker Runtime，可直接进入本地镜像并挂载当前工作目录（镜像标签和 `ASCEND_VISIBLE_DEVICES` 按现场实际分配调整）：

```bash
IMAGE=ghcr.io/dkuav/luciole-ascend:2.10.0.post7-cann9.2.0-910b-ubuntu22.04-py3.12
docker image inspect "$IMAGE" >/dev/null
docker run --pull=never --rm -it --runtime=ascend -e ASCEND_VISIBLE_DEVICES=0 \
  -v "$PWD:/work" -w /work --entrypoint /bin/bash "$IMAGE" -l
```

进入容器后先跑一遍 `bash scripts/probe_env.sh`，确认环境正常再执行编译和推理：

```bash
SOC_910B=Ascend910B1  # 替换为实际芯片完整型号
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_910B" \
  --output-dir work/910b-float
bash scripts/run_model.sh --model work/910b-float/model.om --device 0 \
  --input images=work/input.npy --output-dir work/910b-float-outputs \
  --reference work/reference
```

**常用选项与产物说明**：
- **编译参数**：`compile.sh` 固定使用 `--framework=5`（ONNX）。动态输入或覆盖形状可传 `--input-shape 'images:1,3,640,640'`，其他原生 ATC 参数可通过重复传入 `--atc-arg=--参数名=参数值` 透传。每次编译需指定新的输出目录；目录内会留存 `exit-code.txt`（外部工具退出码）、`tool.log`（完整日志）和 `command.txt`（可直接重放的命令）。Shell 脚本会保留路径中的空格，但 ATC、AMCT 及其配置文件本身仍受厂商字符集限制，建议路径尽量只使用 ASCII 无空格字符。
- **推理与名称映射**：`run_model.sh` 会保存 OM 的原始输出张量。传入 `--reference` 时按张量名逐一对比；若 ATC 重命名了输出节点，可用 `--output-map 'ONNX输出名=OM输出名'` 显式映射。若输入名发生变化，根据 runner 打印的名称差异改用 OM 实际输入名传入同一份文件即可。
- **精度对比规则**：浮点输出按 `abs(actual - ref) <= atol + rtol * abs(ref)` 判定，默认 `--atol 0.001 --rtol 0.01` 仅作排障基线，正式验收请结合业务指标设定阈值。允许两侧浮点位宽不同（报告会同时列出两端 dtype）；整数和布尔类型按逐元素精确相等校验。若不匹配或超出阈值，命令返回非零退出码；已保存的原始输出可随时独立重比。

### 用 ais_bench 的 Python API 执行 OM

除默认 `pyACL` 后端外，也支持使用昇腾 `ais_bench` 推理包及其配套 `aclruntime` 执行 OM。离线环境需提前按设备上的 CANN 版本、CPU 架构和 Python ABI 备好对应 wheel（例如 ARM64 + Python 3.12 对应 `cp312`、`linux_aarch64` 的 `aclruntime`），运行前加载对应的 CANN `set_env.sh`。 [官方安装与 wheel 清单](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/README.md)

```bash
atc_tools om run-ais-bench --model work/910b-float/model.om --device 0 \
  --input images=work/input.npy --output-dir work/910b-ais-bench-outputs
atc_tools om compare --reference work/reference \
  --actual work/910b-ais-bench-outputs
```

该命令执行一次固定 shape 推理，多输入模型同样可重复传 `--input`（脚本按 OM 输入顺序传给 `InferSession`）。输出沿用 `output_0.npy` 和 `outputs.json` 格式，可直接与 CPU 参考输出或 `pyACL` 输出对比。如需预热、循环推理或吞吐压测，请直接使用官方 `python3 -m ais_bench` 工具。 [官方 Python API](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/API_GUIDE.md)

---

## INT8 量化

### 下载和安装 AMCT

`download_amct.sh` 与 `install_amct.sh` 均接收一个 CANN 版本号参数。下载脚本会从昇腾社区软件源拉取包含 `amct_onnx` 和 `amct_acl` 的 ARM64 / x86_64 压缩包，以及 ONNX Runtime 1.16.0 / 1.20.0 的自定义算子头文件（已验证 `8.5.0`、`9.0.0`、`9.1.0`、`9.2.0` 的包结构）：

```bash
# 在联网机器上下载
bash scripts/download_amct.sh 9.2.0

# 将 work/amct/9.2.0/ 拷贝到目标 Linux 机器后执行安装
source /path/to/amct-venv/bin/activate
bash scripts/install_amct.sh 9.2.0
```

目标机需预装同版本 CANN Toolkit、C++ 编译器和 Python 开发头文件，并以具有 Toolkit 目录写权限的用户执行安装。建议为 AMCT 准备独立的 Python 虚拟环境，预先安装 `pip`、`setuptools` 及下表对应的依赖版本（离线环境可先用 `pip download` 备齐 wheel）。 [华为 AMCT（ONNX）配套表](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/devaids/amct/atlasamct_16_0014.html)

| Python 版本 | 推荐依赖配套 |
| --- | --- |
| Python 3.10 / 3.11 | `onnx==1.14.0`、`onnxruntime==1.16.0`、`numpy==1.24.4`、`protobuf==3.20.3` |
| Python 3.12 | `onnx==1.18.0`、`onnxruntime==1.20.0`、`numpy>=2.0.0`、`protobuf==3.20.3` |

安装脚本会根据目标机架构从 `work/amct/<版本>/` 自动挑选对应安装包，在当前 Python 环境中安装 `amct_onnx` 并编译自定义算子，同时将 `amct_acl` 安装到已加载的 Toolkit 目录，全程支持离线执行。 [华为 AMCT 安装指南](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/amct/atlasamct_16_0025.html)

> 安装脚本会沿用当前已加载的 CANN 环境（或自动加载默认路径）并校验 Toolkit 版本。如需安装到非默认版本或自定义目录，请先手动 `source` 对应的 `set_env.sh`。

### AMCT ONNX 预校准

建议先在联网环境中验证 `amct_onnx calibration --help` 并跑通一次实际校准；若需在离线环境重做校准，请将验证通过的 AMCT 环境一并带入（注意自定义算子库必须与 ONNX Runtime 版本严格配套）。

**1. 准备校准数据与配置**（在已安装 `atc_tools` 的环境中执行）：

挑选能代表真实业务分布的样本，每个校准 batch 单独放一个目录，按 ONNX 输入顺序命名为 `input_0.npy`、`input_1.npy` 等（同一输入在所有 batch 中必须保持 shape、dtype 和预处理一致），并另留独立的验证集。

```bash
atc_tools onnx calibration --model work/model.onnx \
  --sample work/samples/0001 --sample work/samples/0002 --output-dir work/calibration
cp configs/amct-onnx.cfg work/amct-onnx.cfg
```

编辑 `work/amct-onnx.cfg`，将 `batch_num` 设为实际参与校准的 batch 数量；如有敏感层需保持浮点，可添加 `skip_layers: "实际节点名"`。若需更换校准算法或做逐层策略覆盖，请参考对应 AMCT 版本的 `calibration_config_onnx.proto` 扩展配置，并妥善保存最终配置和量化日志。

**2. 执行量化并编译 OM**（激活 AMCT 环境后执行）：

```bash
bash scripts/quantize_model.sh --model work/model.onnx \
  --input-shape 'images:1,3,640,640' --data-types float32 \
  --data-dir work/calibration/input_0 --config work/amct-onnx.cfg \
  --output-dir work/quantized
bash scripts/compile.sh --model work/quantized/model_deploy_model.onnx \
  --soc "$SOC_910B" --output-dir work/910b-int8
```

> `work/calibration/calibration.json` 中已汇总对应的 `input_shape`、`data_types` 和 `data_dir`；多输入模型按清单顺序传入，含分号的参数值在 Shell 中需加引号。量化会产出两份模型：
> - `model_deploy_model.onnx`：供 ATC 编译生成 INT8 OM；
> - `model_fake_quant_model.onnx`：供 CPU 精度仿真。若依赖 AMCT 自定义算子库，给 `atc_tools onnx reference` 传入 `--custom-ops-library /path/to/library.so`，并使用配套版本的 ONNX Runtime。

### ATC 内联校准

若使用支持内联校准的 CANN 环境，请先确认 `amct_acl` 已安装且校准 NPU 对当前进程可见。复制 [ATC 压缩配置模板](../../configs/atc-compression.cfg)，按 `calibration.json` 填写 `input_data_dir` 和 `atc_input_shape`（此处的 shape 包含拼接后的总 batch 维），并将 `infer_soc` 与 `infer_device_id` 设为执行校准的设备；最终生成 OM 的目标芯片仍由 `--soc` 指定：

```bash
cp configs/atc-compression.cfg work/atc-compression.cfg
# 编辑 work/atc-compression.cfg，填入校准数据路径、shape 和校准设备信息
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_910B" \
  --compression-optimize-conf work/atc-compression.cfg --output-dir work/910b-int8-inline
```

如需配置逐层量化策略，可在 `calibration` 段内增加 `config_file` 字段，指向当前 CANN 版本的 AMCT（ACL）简易量化配置。注意内联校准（AMCT-ACL）与 AMCT-ONNX 的配置文件语法不同，请按对应 CANN 版本文档核对样本组织方式及参数约束。 [ATC 压缩配置](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0059.html) · [AMCT ONNX 环境](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/devaids/amct/atlasamct_16_0014.html)

---

## 编译 310 并在目标机验收

进入已确认支持目标芯片及板端 CANN 版本的编译环境。以下以 `Ascend310B4` 为例，请按目标板卡的完整型号替换：

```bash
SOC_310=Ascend310B4
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_310" \
  --output-dir work/310-float
bash scripts/compile.sh --model work/quantized/model_deploy_model.onnx \
  --soc "$SOC_310" --output-dir work/310-int8
```

将浮点版与 INT8 版两份 OM、同一组验证输入及 CPU 参考输出传输到目标 310 主机，沿用第 3 节的 `run_model.sh` 命令分别推理并保存输出。编译目录中的产物与 `exit-code.txt` 用于确认编译通过，目标机上的推理日志与输出用于确认部署通过。如果量化部署 ONNX 与目标 CANN 不兼容，需回到目标版本对应的 AMCT 环境重新量化导出。

**量化精度与性能验收建议**：
- **张量与任务精度**：先在同一台目标设备上对比浮点 OM 与 INT8 OM 的原始输出；对于 YOLO 等视觉任务，应在独立的带标注验证集上使用完全一致的前后处理跑评测脚本，记录两份 OM 的 mAP 及差值，再结合业务容忍度决定是否跳过敏感层。
- **推理时延**：为浮点版和 INT8 版 OM 传入相同的 `--warmup 10 --repeat 50`，对比 `acl.mdl.execute` 的中位数耗时（该统计仅覆盖同步模型执行阶段，不含前后处理）。

---

## YOLO 单图示例

运行单图检测示例前，需准备好 OpenCV、`ais_bench` 与 `aclruntime`，并加载对应的 CANN 环境。工具会自动从 OM 读取输入尺寸和 dtype，适用于单张固定 shape RGB 输入、单个检测输出头的模型。 [ais_bench 安装说明](#用-ais_bench-的-python-api-执行-om)

```bash
atc_tools yolo infer --model work/910b-float/model.om \
  --image work/test.jpg --device 0 --layout v8 --output-dir work/detections
```

根据模型输出张量格式选择 `--layout`：
- `v5`：`[1, N, 5+C]`（含 objectness，执行按类别 NMS）
- `v8`：`[1, 4+C, N]`（执行按类别 NMS）
- `xyxy`：`[1, N, 6]`（直接输出坐标、置信度与类别）

检测结果会写入 `detections.json` 和可视化图片 `annotated.jpg`，框坐标已映射回原图尺度。

如需单独生成 CPU 参考输入，或对已导出的预测张量离线做后处理，可分步调用 `preprocess` 和 `postprocess`：

```bash
atc_tools yolo preprocess --image work/test.jpg \
  --width 640 --height 640 --output work/input.npy
# OM 推理完成后，参考 outputs.json 确认预测张量文件名
atc_tools yolo postprocess --image work/test.jpg \
  --width 640 --height 640 --predictions work/910b-float-outputs/output_0.npy \
  --layout v8 --output-dir work/detections
```

`yolo preprocess` 采用等比缩放加 `114` 填充、转 RGB、浮点除以 `255` 的标准流程，输出 NCHW 张量（默认 `float32`，可通过 `--dtype float16` 切换）。使用前请先核对模型的原始预处理逻辑是否一致。

---

## 排障

| 现象 | 排查方向 |
| --- | --- |
| ATC 提示找不到算子或不支持目标型号 | 核对完整的 `soc_version`、CANN Toolkit 与 `ops` 算子包版本、ONNX opset 及 `tool.log` 中的详细报错。 |
| ATC 可用，但 `pyACL` 导入或指定设备失败 | 依次检查 Python ABI 匹配度、CANN 运行时动态库路径、逻辑设备 ID 及容器 NPU 设备挂载情况。 |
| 校准数据大小不匹配 | 核对多输入顺序、dtype、单 batch shape 以及批次数；注意 ATC 内联校准需使用拼接后的 `atc_input_shape`。 |
| OM 描述的字节数或格式与输入张量不一致 | 检查 ATC 编译时的输入 shape、数据格式及 AIPP 等选项；推理 runner 要求输入为可按 OM 描述还原的连续内存张量。 |
| 推理输出数值误差超限 | 先用完全相同的输入对比浮点 ONNX 与浮点 OM，确认基线无误后再排查量化差异（可分别检查 CPU 仿真输出与 NPU 实机输出）。 |

### AMCT-ONNX 进程被杀

对于单输入图片一键量化，`--batch-num` 控制读取的图片总数，每次校准前向推理的 batch size 仍为 `1`。如果量化进程被杀，建议先用 16 张有代表性的图片配合新输出目录复跑（使用 `--config` 时在配置文件中将 `batch_num` 设为 `16`；昇腾官方文档亦建议从 16 或 32 个校准样本起步）。 [AMCT-ONNX 手工调优](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/devaids/amct/atlasamct_16_0146.html)

排查时先看输出目录中的 `amct-onnx.log`：如果日志最后停在保存 `modified_model.onnx`，通常说明进程是在校准前保存中间模型阶段或刚进入校准时被终止。若确认收到 `SIGKILL`，请在故障时间段检查宿主机内核 OOM 日志，并核对容器或作业调度器的内存上限（cgroup v2 下需同时查看当前 cgroup 与父 cgroup 的 `memory.max`）。 [Linux cgroup v2 内存控制](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory)

```bash
journalctl -k -b --no-pager | grep -Ei 'oom-kill|out of memory|killed process' | tail -15
```

---

## 本地开发检查

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
uv run --locked pytest -q tests
```

本地单元测试基于 ONNX Runtime CPU 和模拟 `pyACL`，用于验证数据格式、接口参数传递与资源释放；真实的 ATC 编译、AMCT 量化与 NPU 设备推理仍需按本文流程在目标昇腾环境中完成验收。
