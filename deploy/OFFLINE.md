# Single-container deployment on Ascend

Pull the tested Linux ARM64 application tag on the connected host. Run `bash deploy/pack-offline.sh atc-bundle "$ATC_APP_IMAGE"` and transfer the bundle and release receipt to the lab.

## Fresh installation

1. Run `sha256sum -c SHA256SUMS` and `docker image load -i images.tar`. Compare the loaded ID and platform with `images.txt`.
2. Copy `lab.env.example` to `lab.env`. Set `ATC_APP_IMAGE` to the imported tag and `ATC_HOST_DATA_DIR` to a dedicated directory on a local Linux filesystem. The application initializes SQLite here and stores assets and logs alongside it. Back up the entire directory with the service stopped.
3. Set `SSH_CONFIG_DIR` to a root-owned directory (0700) containing nonempty root-owned `authorized_keys` (0600). API and SSH bind to loopback by default. Set the bind addresses for the laboratory network. Use HTTPS for secure browser cookies, or disable secure cookies for a loopback HTTP test.
4. To enable device verification, set `ATC_RUNTIME=ascend` and set `ATC_910B2_DEVICE` to the assigned physical device. Configure `ATC_AMCT_TARGETS` for the validated INT8 targets.
5. Run `docker compose --env-file lab.env up -d --no-build --pull never --wait`. Provision credentials with `docker compose --env-file lab.env exec app atc_service set-browser-password lab-user` or issue a bearer token with `atc_service issue-token lab-user` in the container.

## Profiling acceptance

Reserve the assigned 910B2 for the capture window, including SSH users and workloads in other containers. The scheduler serializes inference, device comparisons and profiling on its device slot. Coordinate external workloads with the lab operator. Label any overlap as shared load and repeat the baseline on an idle device. For end-to-end comparisons, avoid heavy host compilation and fix the model, inputs, warm-up and measurement counts.

Run floating-point and INT8 conversions for both targets. On 910B2 run comparison and profiling, submit inference while profiling and verify the device execution intervals do not overlap. Cancel a profile and confirm its children have exited before the next device job starts.

## Offline wheel verification and release

Each application build follows the `ascend-atc` channel in `deploy/base-image.txt`. CI resolves the current channel once and records the exact base used in the application's `ATC_BASE_IMAGE` environment variable and the release receipt's `base-image` file. To deploy an upgrade offline, import the newly tested application image and recreate the container.

For the first upgrade from the separate `atc-service` distribution, deploy the new image so the container launcher uses the new `atc_service` command. Subsequent wheel updates can use the procedure below.

Drain active jobs. Verify the candidate wheel's SHA-256 and install it over SSH using `/opt/onnx/bin/python -I -m pip install --no-index --no-deps --force-reinstall ./PACKAGE.whl`. New execution processes load the installed tools. Restart the same container if resident service code changed. Record the original reproducer, result and wheel checksum.

Use the validated candidate's CI run for release. Import the tested application image, update `ATC_APP_IMAGE`, and recreate the container with the same data and SSH mounts. Verify readiness, the original reproducer and a historical download. Retain the source commit, wheel checksum, base digest and final registry digest.

Image tags use `<version>-<short commit SHA>`. Candidate images use the Hatch VCS wheel version, for example `0.1.1.dev6-2c63f75`. Release images use the Git release tag as the version prefix, for example `v0.2.0-2c63f75`. CI generates the short SHA with `git rev-parse --short=7`, which uses more characters when needed to identify the commit unambiguously. An existing image tag blocks publication.

For release operators, select the candidate run from **ARM64 service CI** and verify its source commit and wheel checksum. Create the release tag on that exact commit. Dispatch the workflow at the tag with the numeric candidate run ID as the `wheel_run` input. CI uses that wheel for the tested image and attaches the receipt to the GitHub release. Ordinary successful development builds publish a unique candidate tag and a release-receipt artifact.
