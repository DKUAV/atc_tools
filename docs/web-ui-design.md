# ATC Tools Web UI Design

Serve the private workbench from the existing FastAPI service on the same origin as its browser API. Use local HTML, CSS, and JavaScript so the UI runs on the laboratory intranet. The backend owns authorization, capability discovery, workflow construction, and durable task state. [Web UI requirements](web-ui-requirements.md) · [Service architecture](service-design.md)

## Browser screens and data flow

1. The entry page signs the user in to a server-side session. The task list loads account-owned summaries. The conversion form loads available targets and presets from the server.
2. Selecting a local ONNX triggers an upload and model inspection. The UI waits for the result, displays model inputs, and enables valid choices. It presents a read-only plan from the chosen workflow preset and requested options.
3. Calibration and validation each use one upload control with file and folder commands. The UI presents one common dimensions configuration and one common numerical configuration when either dataset is needed.
4. The browser submits the selected preset, required resources, shape selections, and shared preparation settings. It keeps one idempotency key for a submission attempt and opens the returned task or workflow ID. Poll status and read logs by offset. Use the backend's step status to render the indicators.
5. The detail page lists artifacts as they are published. The history screen continues to display state and logs after the files have been removed from the session.

## Browser API and validation

Keep the existing bearer-token API for scripts. The browser uses same-origin endpoints with server-side sessions and CSRF protection. The browser's conversion entry point accepts preset IDs and structured workflow inputs. Expose status, logs, and downloads as task resources. Model inspection is part of the upload flow. The browser displays supported workflows based on the capability response so an unavailable executor stays out of the submission form.

Validate both the browser request schema and the operation's prerequisites in the service. Use resource IDs throughout the request. Store the resolved preset and preprocessing settings with the workflow so retries use the same interpretation. Derive the workflow steps on the server from the enabled target, preset, and optional checks. Keep the existing isolated task entry points as the execution boundary.

## Session ownership and cleanup

Use a signed-in account and a server-issued session identifier. Scope upload and download permissions for temporary files to that session. Persist jobs and logs by account. Browser logout revokes the session, and a housekeeping process enforces server-side expiry even when a browser close event never arrives.

Use stored references from active jobs or workflows to hold input files and intermediate outputs until their final use. Make expired files unavailable immediately and delete their bytes when unreferenced. Keep resource markers that explain why a download is unavailable in history. Scope cleanup to service-managed paths, make retries idempotent, and retain attempt logs separately from temporary artifacts.

## Input preparation contract

For each model input, resolve input name, declared shape, dtype, and spatial axes where applicable. Map datasets to samples with named fields. Decode the input according to its format. Apply the common dimensions operation first, then the selected numerical transformation, and then validate the final tensor. Presets specify layout, padding, and any normalization parameters needed for reproducibility. A custom numeric configuration uses finite means and strictly positive standard deviations matched to the relevant channels.

For 910B2 comparison, prepare validation tensors once and reuse them for the relevant CPU and NPU executions. Profiling references the primary compiled OM and the requested validation inputs. Keep the individual stage outcomes visible so users can identify which result is available.

Each accepted conversion extends the existing job with a snapshot of its resolved workflow and ordered dataset file IDs. The worker keeps the input mount read-only, reads fixed stage snapshots from the executor, and publishes allowlisted outputs when each stage completes. The primary OM remains downloadable when optional verification fails. Simulation uses the same preparation path and marks all generated conversion or NPU artifacts as simulated.

## Implementation and verification order

1. **Connect the first workbench flow.** Serve the private UI alongside FastAPI. Add browser sign-in, model upload and inspection, capability-driven compile, job history, status, logs, and downloads. Scope files to the session and clear them after dependent jobs finish. Verify the flow with the existing simulated 310B4 preset and real ONNX inspection.
2. **Extend the real workflows.** Extend the worker and request schemas for dataset intake, shared tensor preparation, INT8, and 910B2 checks. Publish each capability after verifying its executor and target preset.
3. **Verify the complete operator experience.** Exercise success, failed inspection or compilation, duplicate submit, cancellation, session expiry, and artifact cleanup. Test representative narrow and wide layouts and reduced motion. On the laboratory host, verify the enabled real toolchains and NPU operations.
