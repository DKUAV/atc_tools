# Single-Container Offline Deployment on Ascend

[English](README.md) | [中文](README_zh.md)

This guide covers packing a tested Linux ARM64 application image for air-gapped Ascend lab hosts, fresh installation, 910B2 profiling acceptance, and offline upgrades/releases.

## 1. Prepare the Offline Bundle (Connected Host)

Pull the tested Linux ARM64 application image on an internet-connected machine and run the packing script:

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

The script writes `compose.yaml`, `lab.env.example`, `README.md`, `README_zh.md`, `images.txt`, `images.tar`, and `SHA256SUMS` into `atc-bundle/`. Transfer the entire `atc-bundle/` directory and the corresponding release receipt to the offline host.

## 2. Fresh Installation (Offline Host)

The commands below assume you are inside the transferred `atc-bundle/` directory. If running from the repository root instead, add `-f deploy/compose.yaml` to `docker compose` commands.

### 2.1 Verify and Load the Image

```bash
sha256sum -c SHA256SUMS
docker image load -i images.tar
```

Verify that the loaded image ID and platform (`linux/arm64`) match `images.txt`.

### 2.2 Configure Environment Variables and Permissions

Copy the template and edit `lab.env`:

```bash
cp lab.env.example lab.env
```

All environment variables supported by `compose.yaml` are listed below (items marked **Required** must be set in `lab.env`):

| Variable | Default | Description |
| --- | --- | --- |
| `ATC_APP_IMAGE` | **Required** | Exact image tag recorded in `images.txt`. |
| `SSH_CONFIG_DIR` | **Required** | Mounted at `/root/.ssh:ro`. Must be a root-owned directory (`0700`) containing a nonempty root-owned `authorized_keys` (`0600`). |
| `ATC_HOST_DATA_DIR` | `/srv/atc-tools` | Mounted at `/data`; must reside on a local Linux filesystem. Stores SQLite, model assets, and logs. **Stop the service before backing up this directory.** |
| `ATC_API_BIND` | `127.0.0.1` | Host bind address for HTTP API / Web Workbench; adjust for lab network access. |
| `ATC_API_PORT` | `8766` | Host port mapped to container port `8080`. |
| `ATC_SSH_BIND` | `127.0.0.1` | Host bind address for SSH; adjust for lab network access. |
| `ATC_SSH_PORT` | `2222` | Host port mapped to container port `22`. |
| `ATC_BROWSER_SECURE_COOKIES` | `1` | Enable HTTPS-only session cookies. Keep `1` behind an HTTPS reverse proxy; set `0` only for local loopback HTTP testing. |
| `ATC_BROWSER_SESSION_SECONDS` | `43200` | Browser session lifetime in seconds (default 12 hours). |
| `ATC_RUNTIME` | `runc` | Container runtime. Set to `ascend` to enable 910B2 inference and profiling. |
| `ATC_910B2_DEVICE` | *(empty)* | Assigned physical 910B2 device ID (also passed to `ASCEND_VISIBLE_DEVICES`). |
| `ATC_AMCT_TARGETS` | *(empty)* | Comma-separated list of validated INT8 targets (e.g. `310B4,910B2` in `lab.env.example`). |
| `ATC_SIMULATE_COMPILE` | `0` | Set `1` to enable hardware-free simulation mode; keep `0` in production. |
| `ATC_CPUS` | `4` | Container CPU core limit (`cpus`). |
| `ATC_MEMORY_BYTES` | `8589934592` | Container memory limit in bytes (default 8 GiB, `mem_limit`). |
| `ATC_CPU_SLOTS` | `2` | Concurrent slots for CPU jobs (e.g. ONNX inspection, CPU inference). |
| `ATC_COMPILE_SLOTS` | `1` | Concurrent slots for compilation/quantization jobs. |
| `ATC_TIMEOUT_SECONDS` | `900` | Per-job execution timeout in seconds (default 15 minutes). |
| `ATC_MAX_PENDING_JOBS` | `8` | Maximum queued pending jobs. |
| `ATC_MAX_UPLOAD_BYTES` | `536870912` | Max single upload size in bytes (default 512 MiB). |
| `ATC_MAX_LOG_BYTES` | `4194304` | Max log size per job in bytes (default 4 MiB). |
| `ATC_MAX_OUTPUT_BYTES` | `1073741824` | Max artifact output size per job in bytes (default 1 GiB). |
| `ATC_MAX_ACCOUNT_BYTES` | `4294967296` | Total storage quota per account in bytes (default 4 GiB). |
| `ATC_SCRATCH_BYTES` | `4294967296` | Scratch workspace size limit in bytes (default 4 GiB). |

### 2.3 Start the Service and Provision Credentials

```bash
# Start container and wait for /readyz healthcheck
docker compose --env-file lab.env up -d --no-build --pull never --wait

# Set password for a Web Workbench account
docker compose --env-file lab.env exec app atc_service set-browser-password lab-user

# Issue an API Bearer Token if needed (not required for Web Workbench only)
docker compose --env-file lab.env exec -T app atc_service issue-token lab-user
```

## 3. Profiling Acceptance

Before running 910B2 profiling (`msprof`), observe the following isolation rules:

- Reserve exclusive access to the assigned 910B2 for the capture window, excluding SSH users and workloads in other containers.
- The internal scheduler serializes inference, device comparison, and profiling on its device slot, but external workloads must be coordinated with the lab operator.
- If any external overlap occurs during capture, label the run as shared load and repeat the baseline on an idle device.
- For end-to-end comparisons, avoid heavy host compilation and keep the model, inputs, warm-up count, and measurement count fixed.

Recommended acceptance checks:

1. Run floating-point and INT8 conversions for both `310B4` and `910B2`.
2. Run accuracy comparison and profiling on `910B2`; submit an inference job while profiling is active and verify their device execution intervals do not overlap.
3. Cancel an in-progress profiling job and confirm all child processes exit before the next device job starts.

## 4. Offline Upgrades, Wheel Verification, and Releases

### 4.1 Image Upgrades

Each application build follows the `ascend-atc` base image channel in `deploy/base-image.txt`. CI resolves the channel once and records the exact base image in the container's `ATC_BASE_IMAGE` environment variable and the release receipt's `base-image` file.

To upgrade offline, import the newly tested application image, update `ATC_APP_IMAGE` in `lab.env`, and recreate the container with the existing data directory and SSH mounts.

> **Upgrading from legacy standalone `atc-service`**: The first upgrade must deploy the new image so the container entrypoint switches to the `atc_service` command. Subsequent wheel patch verifications can use the hot-install procedure below.

### 4.2 Candidate Wheel Verification (via SSH)

To test a candidate wheel without replacing the full image:

1. Drain active jobs.
2. Verify the candidate wheel's SHA-256 checksum.
3. Install over SSH inside the container:
   ```bash
   /opt/onnx/bin/python -I -m pip install --no-index --no-deps --force-reinstall ./PACKAGE.whl
   ```
4. New execution subprocesses load the updated tools automatically; restart the container if resident service code changed.
5. Record the original reproducer, verification result, and wheel checksum.

### 4.3 Production Releases

- **Release operators**:
  1. Select the validated candidate run in **ARM64 service CI** and verify its source commit and wheel checksum.
  2. Create the release tag on that exact commit.
  3. Dispatch the release workflow on that tag with the numeric candidate run ID in `wheel_run`. CI reuses that wheel to build and test the release image and attaches the receipt to the GitHub Release. (Ordinary development builds only publish a unique candidate tag and a `release-receipt` artifact.)
- **Offline production rollout**:
  1. Import the final tested application image, update `ATC_APP_IMAGE` in `lab.env`, and recreate the container with the same data and SSH mounts.
  2. Verify `/readyz`, the original reproducer, and downloading a historical artifact.
  3. Archive the source commit, wheel checksum, base image digest, and final registry digest.
