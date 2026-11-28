# Offline ATC Tools deployment

## Prepare the bundle

Use a connected Linux ARM64 Docker host. Check out the release tag used by the published service image so the CPU and CANN executors use the same source. The pinned `ascend-atc` base in the MobileNetV2 workflow contains CANN 8.5.0, the 910B and 310B operator packages, and AMCT. Build one CANN executor and use it for both targets.

Set `ATC_RELEASE_IMAGE` to the published service image, such as `ghcr.io/dkuav/atc-tools-service:v0.1.0`. Set `ASCEND_ATC_IMAGE` to the verified base reference from the MobileNetV2 workflow and put a disposable URL-safe `ATC_DB_PASSWORD` in `build.env` for Compose interpolation.

```bash
docker pull --platform linux/arm64 "$ATC_RELEASE_IMAGE"
docker image tag "$ATC_RELEASE_IMAGE" atc-service:local
docker compose --env-file build.env -f deploy/compose.yaml --profile build build executor
docker compose --env-file build.env -f deploy/compose.yaml pull postgres
docker build --platform linux/arm64 -f deploy/Dockerfile.ascend-executor \
  --build-arg ASCEND_ATC_IMAGE="$ASCEND_ATC_IMAGE" -t atc-executor:ascend .
bash deploy/pack-offline.sh atc-lab-bundle atc-executor:ascend
```

The pack command checks that the images share a platform and includes an image ID and platform inventory, Compose, a blank configuration template, and SHA-256 checksums. Omit the CANN executor argument to create the simulator bundle used by the `offline` check in the **MobileNetV2 Ascend compilation** workflow. Create credentials on the destination host.

## Install on the offline server

Transfer the bundle, verify it, and load the images.

```bash
sha256sum -c SHA256SUMS
docker image load -i images.tar
cp lab.env.example lab.env
```

Fill `ATC_DB_PASSWORD` with a fresh URL-safe password and set `ATC_HOST_DATA_DIR` to the verified host storage path. If the configured inspection image uses a custom tag, set it to the matching entry in `images.txt`.

- For the full bundle, set both `ATC_COMPILE_IMAGE` and `ATC_COMPILE_910B2_IMAGE` to the CANN executor tag in `images.txt`, such as `atc-executor:ascend`. Set `ATC_AMCT_TARGETS` and `ATC_910B2_DEVICE` after confirming those capabilities on the laboratory host. The MobileNetV2 compile workflow uses `ATC_MEMORY_BYTES=8589934592` per conversion executor. Size this limit and compile concurrency for the laboratory host.
- For a simulator bundle, set `ATC_SIMULATE_COMPILE=1`. Leave `ATC_COMPILE_IMAGE`, `ATC_COMPILE_910B2_IMAGE`, `ATC_AMCT_TARGETS`, and `ATC_910B2_DEVICE` blank.

Place the service data directory on a dedicated volume or apply a host filesystem quota. The service also limits account storage, outstanding jobs, per-task memory, output bytes and temporary workspace. Configure the laboratory HTTPS proxy to forward the loopback endpoint and apply login rate and upload size limits. Keep secure cookies enabled.

```bash
docker compose --env-file lab.env -f compose.yaml up -d --no-build --pull never
curl --fail --show-error --retry 30 --retry-all-errors --retry-delay 1 \
  --max-time 5 http://127.0.0.1:8766/readyz
docker compose --env-file lab.env -f compose.yaml exec api atc-service set-browser-password lab-user
```

`/healthz` checks PostgreSQL. `/readyz` also checks the worker heartbeat, configured local images and, when enabled, the `ascend` runtime registration. On the lab host, run one job for each configured target and preset. Confirm AMCT calibration for each enabled INT8 target, actual output comparison and `msprof` for 910B2, non-root device access, artifact checksums and session cleanup. Preserve the image inventory, driver and CANN versions with the acceptance results.

Stop a deployment using `docker compose --env-file lab.env -f compose.yaml down`. Retain its PostgreSQL volume and host data when upgrading. Finish active workflows before replacing the API and worker images.
