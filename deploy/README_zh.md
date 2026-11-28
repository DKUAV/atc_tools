# 昇腾单容器离线部署指南

[English](README.md) | [中文](README_zh.md)

本文档用于将已测试的 Linux ARM64 应用镜像打包并部署到无外网环境的昇腾实验室主机，涵盖离线打包、首次安装、910B2 性能采集验收，以及离线升级与版本发布流程。

## 1. 准备离线包（联网环境）

在联网主机上拉取测试通过的 Linux ARM64 应用镜像，执行打包脚本：

```bash
# Linux / macOS (Bash)
export ATC_APP_IMAGE="<tested-arm64-image-tag>"
docker pull "$ATC_APP_IMAGE"
bash deploy/pack-offline.sh --output-dir atc-bundle --image "$ATC_APP_IMAGE"
```

```powershell
# Windows (PowerShell)
$env:ATC_APP_IMAGE = "<tested-arm64-image-tag>"
docker pull --platform linux/arm64 $env:ATC_APP_IMAGE
pwsh deploy/pack-offline.ps1 -OutputDir atc-bundle -Image $env:ATC_APP_IMAGE
```

打包脚本会将 `compose.yaml`、`lab.env.example`、`README.md`、`README_zh.md`、`images.txt`、`images.tar` 及校验文件 `SHA256SUMS` 输出到 `atc-bundle/` 目录。将整个 `atc-bundle/` 目录及对应的发布回执（release receipt）传输到离线主机。

## 2. 首次安装（离线主机）

以下命令默认在解压后的离线包目录（`atc-bundle/`）内执行；如在仓库根目录执行，请为 `docker compose` 命令加上 `-f deploy/compose.yaml`。

### 2.1 校验并导入镜像

```bash
sha256sum -c SHA256SUMS
docker image load -i images.tar
```

导入完成后，核对镜像 ID 和平台架构（`linux/arm64`）与 `images.txt` 中的记录一致。

### 2.2 配置环境变量与目录权限

复制环境模板并编辑 `lab.env`：

```bash
cp lab.env.example lab.env
```

`compose.yaml` 支持的全部环境变量如下（带 `必填` 标记的项必须在 `lab.env` 中显式配置）：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `ATC_APP_IMAGE` | **必填** | 填写 `images.txt` 中对应的精确镜像标签。 |
| `SSH_CONFIG_DIR` | **必填** | 挂载为容器 `/root/.ssh:ro`。需为 root 所有的目录（权限 `0700`），目录内包含非空、root 所有的 `authorized_keys`（权限 `0600`）。 |
| `ATC_HOST_DATA_DIR` | `/srv/atc-tools` | 挂载为容器 `/data`，需指向本地 Linux 文件系统上的专用目录。应用会在此初始化 SQLite 数据库并存放模型资产与日志。**备份该目录前必须先停止服务。** |
| `ATC_API_BIND` | `127.0.0.1` | HTTP API / Web 工作台宿主机绑定地址，实验室网络访问请按需调整。 |
| `ATC_API_PORT` | `8766` | HTTP API / Web 工作台宿主机端口（映射到容器 `8080`）。 |
| `ATC_SSH_BIND` | `127.0.0.1` | SSH 宿主机绑定地址，实验室网络访问请按需调整。 |
| `ATC_SSH_PORT` | `2222` | SSH 宿主机端口（映射到容器 `22`）。 |
| `ATC_BROWSER_SECURE_COOKIES` | `1` | 是否启用安全 Cookie。生产/内网请通过 HTTPS 反向代理访问并保持 `1`；仅本机回环 HTTP 测试时可设为 `0`。 |
| `ATC_BROWSER_SESSION_SECONDS` | `43200` | 浏览器登录会话有效期（秒），默认 12 小时。 |
| `ATC_RUNTIME` | `runc` | 容器运行时。启用 910B2 设备推理与性能分析时设为 `ascend`。 |
| `ATC_910B2_DEVICE` | 空 | 分配给容器的 910B2 物理设备 ID（同时透传给 `ASCEND_VISIBLE_DEVICES`）。 |
| `ATC_AMCT_TARGETS` | 空 | 已完成验证的 INT8 目标芯片列表，逗号分隔（`lab.env.example` 示例为 `310B4,910B2`）。 |
| `ATC_SIMULATE_COMPILE` | `0` | 设为 `1` 开启无昇腾硬件的模拟模式；真实部署保持 `0`。 |
| `ATC_CPUS` | `4` | 容器可用 CPU 核数上限（`cpus`）。 |
| `ATC_MEMORY_BYTES` | `8589934592` | 容器内存上限（字节，默认 8 GiB，`mem_limit`）。 |
| `ATC_CPU_SLOTS` | `2` | CPU 类任务（如 ONNX 检查、CPU 推理）的并发槽位数。 |
| `ATC_COMPILE_SLOTS` | `1` | 编译/量化类任务的并发槽位数。 |
| `ATC_TIMEOUT_SECONDS` | `900` | 单个任务的执行超时时间（秒，默认 15 分钟）。 |
| `ATC_MAX_PENDING_JOBS` | `8` | 允许排队等待的最大任务数。 |
| `ATC_MAX_UPLOAD_BYTES` | `536870912` | 单个上传文件大小上限（字节，默认 512 MiB）。 |
| `ATC_MAX_LOG_BYTES` | `4194304` | 单个任务日志大小上限（字节，默认 4 MiB）。 |
| `ATC_MAX_OUTPUT_BYTES` | `1073741824` | 单个任务输出产物大小上限（字节，默认 1 GiB）。 |
| `ATC_MAX_ACCOUNT_BYTES` | `4294967296` | 单个账号的总存储配额上限（字节，默认 4 GiB）。 |
| `ATC_SCRATCH_BYTES` | `4294967296` | 任务临时工作区空间上限（字节，默认 4 GiB）。 |

### 2.3 启动服务并初始化账号

```bash
# 启动容器并等待 /readyz 健康检查通过
docker compose --env-file lab.env up -d --no-build --pull never --wait

# 为 Web 工作台设置账号密码
docker compose --env-file lab.env exec app atc_service set-browser-password lab-user

# 按需为 HTTP API 签发 Bearer Token（仅使用 Web 工作台时无需执行）
docker compose --env-file lab.env exec -T app atc_service issue-token lab-user
```

## 3. 性能采集验收（Profiling）

进行 910B2 性能采集（`msprof`）前，请确认以下环境隔离要求：

- 为目标 910B2 预留独占时间窗口，避免 SSH 用户或其他容器同时占用设备。
- 服务内部调度器会在同一设备槽位上串行化推理、设备精度对比与性能采集，但无法隔离容器外部任务，需与实验室运维人员协调。
- 若采集期间存在外部重叠负载，应将结果标注为共享负载，并在设备空闲时重跑基线。
- 进行端到端性能对比时，避免宿主机重度编译干扰，并固定模型、输入样本、预热次数和测量次数。

建议的验收检查项：

1. 对 310B4 和 910B2 两个目标分别执行浮点与 INT8 转换。
2. 在 910B2 上运行精度对比与性能采集；在性能采集执行期间提交推理任务，确认两者的设备执行区间不重叠。
3. 取消一次正在执行的性能采集任务，确认其子进程已全部退出后，下一个设备任务才开始启动。

## 4. 离线升级、Wheel 验证与发布

### 4.1 镜像升级

每次应用镜像构建都会跟踪 `deploy/base-image.txt` 中的 `ascend-atc` 基础镜像通道。CI 会解析当前通道一次，并将实际使用的基础镜像记录在应用容器的 `ATC_BASE_IMAGE` 环境变量以及发布回执的 `base-image` 文件中。

离线升级时，导入新测试通过的应用镜像，更新 `lab.env` 中的 `ATC_APP_IMAGE`，沿用原有的数据目录和 SSH 挂载重建容器即可。

> **从旧版独立 `atc-service` 迁移注意**：首次升级必须直接部署新镜像，确保容器启动入口切换到新的 `atc_service` 命令；完成此次切换后，后续的 wheel 补丁验证才可使用下文的热更新步骤。

### 4.2 候选 Wheel 离线验证（通过 SSH）

如需在不替换整包镜像的前提下快速验证候选 wheel：

1. 排空服务中正在运行的任务。
2. 校验候选 wheel 文件的 SHA-256 校验和。
3. 通过 SSH 登录容器执行覆盖安装：
   ```bash
   /opt/onnx/bin/python -I -m pip install --no-index --no-deps --force-reinstall ./PACKAGE.whl
   ```
4. 新启动的任务子进程会自动加载更新后的工具；如果常驻服务代码有变更，重启当前容器即可。
5. 记录原始复现用例、验证结果和 wheel 校验和。

### 4.3 正式发布与生产上线

- **发布操作员流程**：
  1. 在 **ARM64 service CI** 中选定已验证的候选运行记录，核对其源码 commit 与 wheel 校验和。
  2. 在该精确 commit 上创建发布标签（release tag）。
  3. 在该 tag 上手动触发发布工作流，将候选运行记录的数字 ID 填入 `wheel_run` 参数。CI 会复用该 wheel 构建并测试最终镜像，并将发布回执附加到 GitHub Release。（普通开发分支构建成功后仅发布唯一的候选 tag 和 `release-receipt` 制品。）
- **离线主机正式上线**：
  1. 导入最终测试通过的应用镜像，更新 `lab.env` 中的 `ATC_APP_IMAGE`，保持原有数据目录和 SSH 挂载不变重建容器。
  2. 验证服务就绪状态（`/readyz`）、原始复现用例以及历史产物的正常下载。
  3. 归档记录本次发布的源码 commit、wheel 校验和、基础镜像 digest 和最终镜像仓库 digest。
