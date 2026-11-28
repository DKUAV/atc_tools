# Offline ATC Tools deployment

## Prepare the bundle

Use a connected Docker host matching the laboratory server's CPU architecture. Build the service and CPU executor from the repository and pull the pinned PostgreSQL image from the Compose configuration. Build each enabled CANN executor using `deploy/Dockerfile.ascend-executor` and a verified target-specific base image.

The `ascend-atc` image definition in the companion images repository supplies an ARM64 CANN 8.5.0 base for 910B2. Prepare a separate compatible 310B4 base with the matching 310B operator package. Each base needs the CANN toolkit and the `/opt/onnx` Python environment used by `deploy/ascend-executor.sh`. For INT8, it also needs AMCT and its compatible Python environment. Check the vendor package requirements for the exact CANN version. [ATC target requirements](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/atctool/atlasatcparam_16_0036.html)

```bash
docker compose --env-file build.env -f deploy/compose.yaml --profile build build api executor
docker compose --env-file build.env -f deploy/compose.yaml pull postgres
docker build -f deploy/Dockerfile.ascend-executor \
  --build-arg ASCEND_ATC_IMAGE="$BASE_310B4" -t atc-executor:310b4 .
docker build -f deploy/Dockerfile.ascend-executor \
  --build-arg ASCEND_ATC_IMAGE="$BASE_910B2" -t atc-executor:910b2 .
bash deploy/pack-offline.sh work/atc-lab-bundle atc-executor:310b4 atc-executor:910b2
```

Set `BASE_310B4` and `BASE_910B2` to the corresponding local image tags. Use a build-time environment file with a disposable URL-safe `ATC_DB_PASSWORD` to satisfy Compose interpolation. The pack command checks that the images share a platform and includes an image ID and platform inventory, Compose, a blank configuration template, and SHA-256 checksums. Omitting both CANN images creates a bundle for a simulator rehearsal. Create credentials on the destination host.

## Install on the offline server

Transfer the bundle, verify it, and load the images.

```bash
sha256sum -c SHA256SUMS
docker image load -i images.tar
cp lab.env.example lab.env
```

Fill `ATC_DB_PASSWORD` with a fresh URL-safe password. Set `ATC_HOST_DATA_DIR` to the verified host storage path, and set the executor tags from `images.txt`. Set `ATC_AMCT_TARGETS` and `ATC_910B2_DEVICE` after confirming those capabilities on the laboratory host. Keep the inspection image and password aligned with the Compose environment. If the configured inspection image uses a custom tag, set it to the matching entry in `images.txt`.

Place the service data directory on a dedicated volume or apply a host filesystem quota. The service also limits account storage, outstanding jobs, per-task memory, output bytes and temporary workspace. Configure the laboratory HTTPS proxy to forward the loopback endpoint and apply login rate and upload size limits. Keep secure cookies enabled.

```bash
docker compose --env-file lab.env -f compose.yaml up -d --no-build --pull never
curl --fail http://127.0.0.1:8766/readyz
docker compose --env-file lab.env -f compose.yaml exec api atc-service set-browser-password lab-user
```

`/healthz` checks PostgreSQL. `/readyz` also checks the worker heartbeat, configured local images and, when enabled, the `ascend` runtime registration. On the lab host, run one job for each configured target and preset. Confirm AMCT calibration for each enabled INT8 target, actual output comparison and `msprof` for 910B2, non-root device access, artifact checksums and session cleanup. Preserve the image inventory, driver and CANN versions with the acceptance results.

Stop a deployment using `docker compose --env-file lab.env -f compose.yaml down`. Retain its PostgreSQL volume and host data when upgrading. Finish active workflows before replacing the API and worker images.
