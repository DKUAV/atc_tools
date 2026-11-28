# ATC Tools 服务

[English](README.md) | [中文](README_zh.md)

ATC Tools Service 提供浏览器工作台（Web Workbench）和基于 Bearer Token 的 HTTP API，用于完成 ONNX 模型检查、OM 模型编译、INT8 量化、推理、精度对比和性能分析（Profiling）。

> **初次使用？** 建议先看[快速开始（模拟模式）](#快速开始模拟模式)，无需昇腾硬件即可在本地体验完整流程。  
> 服务架构、职责划分与执行边界见[服务架构设计](../docs/service-design.md)。

## 目录

- [快速开始（模拟模式）](#快速开始模拟模式)
- [使用浏览器工作台](#使用浏览器工作台)
- [接入真实昇腾工具链](#接入真实昇腾工具链)
- [通过 HTTP API 提交任务](#通过-http-api-提交任务)
- [运行冒烟测试](#运行冒烟测试)
- [GitHub Actions 与预构建镜像](#github-actions-与预构建镜像)

---

## 快速开始（模拟模式）

开启模拟模式（`ATC_SIMULATE_COMPILE=1`）后，无需昇腾硬件即可在本地完整体验 310B4 和 910B2 的工作流。

在安装了 Docker 和 Docker Compose 的 Linux 主机上，从**仓库根目录**按以下步骤操作。

### 1. 设置环境变量

```bash
export ATC_DB_PASSWORD="$(openssl rand -hex 24)"
export ATC_HOST_DATA_DIR="$PWD/work/service-data"
export ATC_SIMULATE_COMPILE=1
export ATC_BROWSER_SECURE_COOKIES=0   # 本地 HTTP 调试浏览器登录时需要关闭安全 Cookie
mkdir -p "$ATC_HOST_DATA_DIR"
```

### 2. 准备镜像

根据你的环境二选一：

**方式 A：从 GitHub 拉取现成服务镜像（ARM64 环境推荐）**

`api` 和 `worker` 共用发布在 GitHub Container Registry（`ghcr.io/dkuav/atc-tools-service`）的预构建镜像。先检出对应的版本标签（例如 `v0.1.0`），拉取并打标为 `atc-service:local`，再构建轻量的 CPU 执行器镜像：

```bash
docker pull ghcr.io/dkuav/atc-tools-service:v0.1.0
docker tag ghcr.io/dkuav/atc-tools-service:v0.1.0 atc-service:local

docker compose -f deploy/compose.yaml --profile build build executor
```

**方式 B：从源码本地构建全部镜像（本地开发或非 ARM64 主机）**

如果你需要修改服务代码，或在 x86_64 主机上运行，请直接从源码构建服务镜像与 CPU 执行器镜像：

```bash
docker compose -f deploy/compose.yaml --profile build build api executor
```

### 3. 启动服务

```bash
docker compose -f deploy/compose.yaml up -d --no-build
```

- **默认地址**：`http://127.0.0.1:8766`（可通过 `ATC_API_PORT` 修改宿主机端口）
- **OpenAPI 文档**：`http://127.0.0.1:8766/openapi.json`
- **就绪检查**：`http://127.0.0.1:8766/readyz`

> **Docker Desktop for Windows 说明**：请将 `ATC_HOST_DATA_DIR` 设为 Windows 宿主机挂载路径，将 `ATC_DOCKER_DATA_DIR` 设为 Linux Docker 守护进程看到的同一目录路径。例如：
> - `ATC_HOST_DATA_DIR=D:/atc-tools-data`
> - `ATC_DOCKER_DATA_DIR=/run/desktop/mnt/host/d/atc-tools-data`

### 常用管理命令

```bash
# 停止服务（保留数据库卷和本地数据文件）
docker compose -f deploy/compose.yaml down

# 为用户签发 API Bearer Token
docker compose -f deploy/compose.yaml exec -T api atc-service issue-token alice

# 通过交互式提示吊销 Token
docker compose -f deploy/compose.yaml exec api atc-service revoke-token
```

---

## 使用浏览器工作台

### 1. 创建浏览器账号

通过交互式命令为本地账号设置密码。重置密码会自动结束该账号现有的浏览器会话。

```bash
docker compose -f deploy/compose.yaml exec api atc-service set-browser-password alice
```

随后在浏览器中打开 `http://127.0.0.1:8766/` 即可登录使用。

> **安全说明**：
> - **本地 HTTP 测试**：启动 Compose 前设置 `ATC_BROWSER_SECURE_COOKIES=0`，并仅绑定到回环地址（`127.0.0.1`）。
> - **实验室 / 内网部署**：请通过 HTTPS 提供服务并保持安全 Cookie 开启（默认 `ATC_BROWSER_SECURE_COOKIES=1`）。当 HTTPS 反向代理与服务在同一宿主机上时，保持 Compose API 端口仅监听回环地址，并在入口层对登录接口配置速率限制与请求体大小限制。

### 2. 用小型示例模型体验流程

在模拟模式（`ATC_SIMULATE_COMPILE=1`）下，**310B4** 和 **910B2** 目标均支持 `original`（原始精度）、`mixed Float16`（混合精度）、`preferred FP16`（优先 FP16）和 `INT8` 预设：

- 选择 **INT8** 时会出现校准数据集上传入口。
- 选择 **910B2** 且开启精度对比或 `msprof` 性能分析时，会出现独立的验证数据集上传入口（与校准集共用维度和数值设置）。
- 模拟器会真实执行 ONNX 检查、数据集预处理以及 CPU 参考推理，并将模拟产物明确标记为 `model.simulated`、`accuracy.json` 和 `profiling.zip`。

在仓库根目录运行以下命令，生成一个极简固定尺寸模型（`work/demo.onnx`）和示例输入张量（`work/demo.npy`）：

```bash
mkdir -p work
uv run --locked python - <<'PY'
import numpy as np
from onnx import TensorProto, helper, save

value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
node = helper.make_node("Relu", ["input"], ["output"])
graph = helper.make_graph([node], "demo", [value], [output])
save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), "work/demo.onnx")
np.save("work/demo.npy", np.array([[-1, 2]], dtype=np.float32), allow_pickle=False)
PY
```

在工作台上传 `work/demo.onnx`，需要验证集或校准集时选择 `work/demo.npy` 即可。

### 3. 数据集格式要求

- **支持格式**：图片（解码为 RGB 或灰度图）、NumPy 数组（`.npy`）和原始二进制文件（`.bin`）。
- **布局与数据类型**：NumPy 数组和原始二进制文件遵循所选模型布局；原始二进制文件默认采用模型输入 shape 与小端序 dtype。
- **多输入模型**：每个样本需按模型输入顺序提供所有输入，支持两种组织方式：
  - 使用同级子目录 `input_0/`、`input_1/` 等，各子目录下保持相同样本文件名；
  - 平铺文件并命名为 `sample.input_0.npy`、`sample.input_1.npy` 等形式。
- **数量上限**：每个数据集最多支持 **128 个样本**。

### 4. 会话与文件清理

- 浏览器会话默认有效期为 **12 小时**（可通过 `ATC_BROWSER_SESSION_SECONDS` 调整）。
- 下载完所需产物后请及时退出登录。
- 退出登录或会话过期后，会话文件的访问权限会立即失效；Worker 会在该会话的待执行任务完成后物理删除对应文件。
- 任务历史记录和日志在下次登录后仍然可见。

完整的界面需求与设计见[Web UI 需求文档](../docs/web-ui-requirements.md)和[Web UI 设计文档](../docs/web-ui-design.md)。

---

## 接入真实昇腾工具链

### 1. 构建昇腾执行器镜像

在兼容昇腾的宿主机上，基于本地准备好的 `ascend-atc` 基础镜像构建执行器镜像（基础镜像需包含与目标芯片匹配的 CANN Toolkit 和算子包）：

```bash
docker build -f deploy/Dockerfile.ascend-executor \
  --build-arg ASCEND_ATC_IMAGE=ascend-atc:8.5.0-910b-ubuntu22.04-py3.11 \
  -t atc-executor:910b2 .
```

### 2. 配置环境变量

保持 `ATC_SIMULATE_COMPILE=0`，并在启动 Compose 前配置目标镜像与运行参数：

| 环境变量 | 说明 |
| --- | --- |
| `ATC_SIMULATE_COMPILE` | 设为 `0` 以启用真实昇腾工具链。 |
| `ATC_COMPILE_IMAGE` | **310B4** 执行器镜像（配置后启用 3 种浮点精度预设）。 |
| `ATC_COMPILE_910B2_IMAGE` | **910B2** 执行器镜像（配置后启用 3 种浮点精度预设）。 |
| `ATC_AMCT_TARGETS` | 启用 INT8 量化的目标列表（需先针对各目标验证 AMCT 校准和 ATC 部署转换）。 |
| `ATC_910B2_DEVICE` | 宿主机 910B2 设备编号（例如 `0`），用于开启板端推理、精度对比和性能分析。 |
| `ATC_CPU_SLOTS` | CPU 任务并发槽位数（默认 `2`）。 |
| `ATC_COMPILE_SLOTS` | 编译任务并发槽位数（默认 `1`）。 |
| `ATC_SCRATCH_BYTES` | 每个工作流容器的临时工作空间大小上限。 |

> **宿主机要求**：Docker 守护进程需要配置 `ascend` 运行时，且执行器 UID `10001` 需具备访问指定 NPU 设备和 Profiling 功能的权限。真实工具链和非 root 运行时访问权限需在实验室宿主机上完成验证。

### 3. 精度模式、量化与性能分析说明

- **浮点精度预设**：对应 ATC 的 `--precision_mode_v2` 参数（[ATC 精度模式文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/atctool/atlasatcparam_16_0069.html)）。
- **INT8 量化**：使用 AMCT 默认激活量化算法，校准批次数由上传的校准数据集决定（[AMCT 配置文档](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/amct/atlasamct_16_0163.html)）。
- **精度对比**：使用 CPU ONNX 参考输出进行对比；对于 INT8 模型，还会与同设备上的原始精度 OM 基线进行对比。
- **性能分析（`msprof`）**：使用共享验证集运行主 OM 并打包采集结果（[CANN 8.5 运行时 Profiling 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/profiling/atlasprofiling_16_0011.html)）。查看压缩包内的 `profiling.json` 可获取输入数量、总耗时和采集体积；设备侧详细耗时请查看压缩包内的原生 `msprof` 文件。

### 4. 并发调度、就绪检查与离线部署

- Worker 持有单个协调锁，并维护独立的 **CPU**、**编译** 和 **设备** 队列（已配置的 910B2 设备占用 1 个槽位）。
- 工作流会先完成主 OM（以及所需的浮点基线）编译，再申请设备槽位，两个阶段保持同一个任务 ID。
- 访问 `/readyz` 可检查 Worker 和执行器镜像的可用状态。
- 离线打包与部署流程见[离线包与上线指南](../deploy/OFFLINE.md)。

---

## 通过 HTTP API 提交任务

所有 API 请求均需携带 `Authorization: Bearer <token>` 请求头。传入稳定的 `Idempotency-Key` 请求头可在请求中断时安全恢复同一个任务。完整的字段类型定义见 `GET /openapi.json`。

### 典型调用流程

1. **查询能力**：调用 `GET /v1/capabilities`，获取旧版任务预设、`workflow_presets` 和 `standalone_operations` 列表。
2. **上传模型**：调用 `POST /v1/assets?filename=model.onnx`（`Content-Type: application/octet-stream`），获取返回的 `input_id`。
3. **上传数据集文件** *（按需）*：调用 `POST /v1/dataset-files?filename=sample.npy`（`application/octet-stream`），将返回的资产 ID 按模型输入顺序组织为 batch 列表。
4. **检查并提交任务**：
   - 先使用 `input_id` 提交 `inspect_onnx` 任务。
   - 检查成功后，可提交 `compile_om` 任务、完整工作流（`POST /v1/workflows`）或独立原子操作（`POST /v1/operations`）。
5. **轮询状态与下载产物**：
   - 查看近期脚本任务：`GET /v1/jobs`
   - 轮询任务状态与产物 ID：`GET /v1/jobs/{job_id}`
   - 增量读取日志：`GET /v1/jobs/{job_id}/logs`（传入上次返回的 `next_offset`）
   - 下载产物：`GET /v1/assets/{asset_id}/download`

### 工作流与重试接口

| 接口 | 请求参数与返回说明 |
| --- | --- |
| `POST /v1/workflows` | `model_id`、`preset`，以及可选的 `calibration`、`validation`、`compare`、`profile`、`shapes`、`preparation`。返回完整转换任务。 |
| `POST /v1/operations` | `operation`、`input_id` 以及下表对应的操作字段。 |
| `POST /v1/jobs/{job_id}/retry` | `step`（`compare` 或 `profile`）以及可选的替换 `validation`。返回关联到原始 OM 的子任务。 |

> 在浏览器工作台中，当可选的验证阶段失败时，界面会提供重试按钮，同时保留原始 OM 可供下载；用户也可以在重试时上传新的验证集文件。

### 独立操作（`POST /v1/operations`）

| `operation` | `input_id` | 其他字段与产出 |
| --- | --- | --- |
| `quantize_onnx` | 已检查的源 ONNX | INT8 `preset` 和 `samples`。产出部署用 ONNX。 |
| `compile_om` | 已检查的 ONNX | `preset`。独立编译 INT8 时，传入量化产出的已检查部署 ONNX。产出 OM 与构建信息。 |
| `run_cpu` | 已检查的 ONNX | `samples`。产出包含 CPU 参考输出的 `tensors.zip`。 |
| `run_npu` | 已检查的源 ONNX | 已启用的 910B2 `preset`、该模型对应的 910B2 OM（`primary_id`）和 `samples`。产出 `tensors.zip`。 |
| `compare_outputs` | 参考张量压缩包 | 第二份张量压缩包 `actual_id`。产出精度对比 JSON 与内联判定结论。 |
| `profile_om` | 已检查的源 ONNX | 已启用的 910B2 `preset`、`primary_id` 和 `samples`。产出 Profiling 压缩包。 |

`POST /v1/operations` 请求示例：

```json
{
  "operation": "run_cpu",
  "input_id": "<inspected-onnx-id>",
  "samples": [["<dataset-asset-id>"]]
}
```

---

## 运行冒烟测试

以下三套冒烟测试均面向本地模拟部署（`ATC_SIMULATE_COMPILE=1`）。

### 1. API 与交付流程冒烟测试

签发两个测试 Token，依次运行 `smoke.py`（上传生成的小型 ONNX、测试双账号隔离、失败日志及模拟产物下载）和 `delivery_smoke.py`（端到端测试独立操作、分阶段日志和失败重试）：

```bash
export ATC_SMOKE_TOKEN_A="$(docker compose -f deploy/compose.yaml exec -T api atc-service issue-token alice)"
export ATC_SMOKE_TOKEN_B="$(docker compose -f deploy/compose.yaml exec -T api atc-service issue-token bob)"

uv run --locked python service/tests/smoke.py
uv run --locked python service/tests/delivery_smoke.py

unset ATC_SMOKE_TOKEN_A ATC_SMOKE_TOKEN_B
```

> 模拟器会返回 `model.simulated` 并标记替代的 NPU 结果。真实 ATC、AMCT 和设备侧行为由实验室验收覆盖。

### 2. 浏览器工作台冒烟测试

在开启模拟模式和本地 HTTP（`ATC_BROWSER_SECURE_COOKIES=0`）的情况下，先通过 `set-browser-password` 配置两个测试账号，再运行 `browser_smoke.py`，验证浏览器接口、混合格式数据集、可选阶段失败、账号隔离与文件清理：

```bash
export ATC_SMOKE_USER_A="alice"
export ATC_SMOKE_PASSWORD_A="<alice-password>"
export ATC_SMOKE_USER_B="bob"
export ATC_SMOKE_PASSWORD_B="<bob-password>"
# 可选：指定宿主机数据目录以验证文件是否已物理删除
export ATC_SMOKE_DATA_DIR="$PWD/work/service-data"

uv run --locked python service/tests/browser_smoke.py
```

---

## GitHub Actions 与预构建镜像

- **ARM64 服务 CI**：在 Pull Request 和推送至 `master` 时自动运行静态检查、单元测试以及上述三套 Compose 模拟测试。推送版本标签（如 `v0.1.0`）时，在通过同样检查后会将服务镜像发布到 `ghcr.io/dkuav/atc-tools-service:v0.1.0`。首次发布需要在 GitHub Packages 中将包可见性设为 public（[GitHub Container Registry 文档](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)）。
- **使用已发布镜像**：见[快速开始中的方式 A](#2-准备镜像)，从 GHCR 拉取指定版本并打标为 `atc-service:local`（`api` 和 `worker` 共用），再基于对应源码标签构建 CPU 执行器。
- **MobileNetV2 昇腾真实编译检查** *（在 Actions 页面手动触发）*：原生 ARM64 CI 会下载并校验固定版本的 [ONNX Model Zoo MobileNetV2 模型](https://github.com/onnx/models/blob/4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177/validated/vision/classification/mobilenet/README.md)，上传确定性校准张量，并验证 310B4 和 910B2 的原始精度与 INT8 OM 下载，同时保留各目标与精度的日志及构建元数据。板端推理与性能分析由实验室验收覆盖。
- **离线包验证**：在同一工作流中选择 `offline`，可验证模拟器离线归档、重新加载 ARM64 镜像并使用 `--pull never` 启动复制的 Compose 配置；选择 `all` 则同时运行两项检查。准备包含共享 CANN 执行器的完整离线包见[离线包与上线指南](../deploy/OFFLINE.md)。
