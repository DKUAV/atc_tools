# ATC Tools Service Requirements

The laboratory intranet service lets users upload ONNX models, convert them to Ascend OM, and download the results. The first milestone delivers this workflow through an authenticated API. The Web UI follows and uses the same API to present supported choices and assemble workflows. Existing tools provide the model preparation, conversion, inference, and comparison operations. [Deployment guide](../guides/deployment-guide.md)

## Users and access

Lab users submit and monitor their own tasks and download their artifacts. Administrators provision accounts and manage available backends, presets, and resource limits. The service manages its own accounts. Browser users sign in with a password, and API clients use revocable tokens associated with the same accounts.

## Conversion and verification workflow

1. Upload a single self-contained ONNX model. All weights are embedded in the file. The service inspects the model and reports input and output names, shapes, and data types so users can identify the required inputs.
2. Choose a target SoC and a compatible conversion preset from the available options. For INT8, upload calibration images and select a supported preprocessing preset. The service checks that the model inputs match that preset.
3. For 910B2, choose whether to run numerical output comparison, collect performance data with `msprof`, or run both. Either option requires validation inputs. Users can reuse the same validation inputs for both options, with a file selected for each required model input.
4. Submit the task and receive a task ID immediately. The task continues after the client disconnects. Users can return to check progress, read logs, and download available results.

| Target | Supported workflow |
| --- | --- |
| Ascend 310B4 | Compile and deliver an OM for 310B4. |
| Ascend 910B2 | Compile and deliver an OM for 910B2. Numerical output comparison and `msprof` collection are independently selectable. |

For a floating-point 910B2 build, the numerical comparison step uses the original ONNX on CPU and the delivered OM on the NPU. When a user requests numerical comparison for an INT8 910B2 build, the task also builds and runs a floating-point OM. It delivers two reports, one comparing the original ONNX with the INT8 OM and one comparing the floating-point OM with the INT8 OM on the same device. All baseline and target runs use the same validation inputs and preprocessing. Each report includes per-output error, the applied tolerances, and the result of the check.

`msprof` runs separately against the primary OM selected for delivery. An unsuccessful optional step or a comparison outside tolerance is visible alongside the compile result. A successfully built OM remains downloadable, and a failed optional step can be retried using that OM.

## API operations and composition

Each operation can be submitted independently. Operations accept resource IDs and supported parameters. The service publishes the available targets, presets, and permitted combinations so both clients and the Web UI can guide selection.

| Operation | Inputs | Results |
| --- | --- | --- |
| Inspect ONNX | Uploaded ONNX model | Model input and output metadata |
| Quantize ONNX | ONNX model, calibration images, preprocessing and quantization presets | ONNX suitable for INT8 compilation |
| Compile OM | ONNX model, target SoC, conversion preset | OM and build information |
| Run CPU inference | ONNX model and validation inputs | Reference output tensors and manifest |
| Run NPU inference | 910B2 OM and validation inputs | Device output tensors and manifest |
| Compare outputs | Two inference results and a tolerance preset | Numerical comparison report |
| Collect performance data | 910B2 OM, validation inputs, profiling preset | Performance summary and collection files |

The API supports resource upload, capability discovery, operation and workflow submission, task status and history, cancellation, step retry, and authorized downloads. Submission returns a durable task ID. Status includes the state of each step, start and end times when available, and an actionable reason for failure or rejection.

The Web UI chooses the steps for a workflow and submits them together. The service validates and persists the dependencies, queues eligible steps, and passes artifact IDs from completed steps to their dependents. The first workflow to support and test is conversion with the target-specific options defined above.

## Safety and user input validation

- Treat user uploads and submitted parameters as untrusted. Verify ownership and allowed actions for every resource, task, log, and artifact read or write.
- Validate file formats, size and count limits, model metadata, and input compatibility. For INT8, check the calibration images and preprocessing preset before starting quantization. Return the field or file that needs correction when validation fails.
- Use approved operation parameters and service-generated paths when invoking tools. Administrators configure toolchain paths, custom libraries, and any extra tool arguments exposed through presets. The service validates the selected target and options again at submission.
- Run work outside the API process with least privilege, a task-specific workspace, access to the task's inputs, and bounded CPU, memory, runtime, and disk use. Apply account-level limits to uploads and outstanding tasks. Limit device access to the resources assigned to each worker.

## Concurrent execution and recovery

Users can upload and submit tasks concurrently. Workers claim persisted steps according to the capacity of the required toolchain and device. Configure compile concurrency separately from inference. Start with one active execution or profiling step per 910B2 device and enforce that capacity across users.

Persist workflow and step states across service restarts. A failed prerequisite blocks its dependent steps. Cancellation releases the reserved resources after the running step stops. Recovery reconciles interrupted work before retrying so that duplicate workers cannot publish competing results for the same step.

## Logging and deliverables

The service logger follows Minifly's `src/minifly/utils/logger.py` for component-scoped logging, configurable levels, console output, and rotating files. Service logs carry request and task IDs where applicable. Record account changes, task actions, and authorization failures for operators. Exclude credentials and tokens from logs.

Capture user-visible execution logs per task. The result view or API response identifies the primary OM, the status of each requested check, and the available reports and profiling files. Build information includes the target SoC, CANN version, selected presets, and checksums for the delivered files. Users can follow a task link to retrieve results, and an example model and request provide a first run for both browser and API users.

## Acceptance criteria

| Area | Acceptance check |
| --- | --- |
| Upload and selection | Upload a small ONNX and inspect its inputs. Verify that models with external weights and files over the upload limit are rejected. Capability discovery returns both targets on a fully configured test deployment. Invalid target and preset combinations are rejected at submission. |
| 310B4 workflow | A submitted conversion produces a downloadable OM and build information. Requests for 910B2-only checks are rejected for this target. |
| 910B2 workflow | Exercise both optional checks individually and together. Verify that INT8 comparison produces both reports and that a failed optional step leaves a compiled OM available. |
| Access control | Two users can each upload, submit, and download their own results. Attempts to use the other user's resource IDs or retrieve their logs or artifacts are denied. A revoked token loses access. |
| Scheduling and recovery | Concurrent submissions return task IDs and respect configured worker and device capacity. Test cancellation, timeouts, duplicate claims, and restart recovery. |
| Logging and delivery | Trace a request to its task through service logs. The task result lists the output files and their checksums, and its log is available only to authorized users. |

In an internet-connected development environment, run upload, API, authorization, and scheduling tests with small synthetic ONNX models and controlled substitutes for CANN commands and devices. Exercise real model inspection and CPU inference there. On the laboratory intranet, verify OM compilation for both targets, 910B2 numerical comparisons, `msprof` output, and enforcement of worker isolation and device limits with the actual toolchains.

## Deployment settings to confirm

Confirm the CANN and AMCT versions and worker placement for each target. Select the initial conversion, preprocessing, tolerance, and profiling presets. Set file size and count limits, task timeouts, worker and account capacity, and retention periods for uploads, artifacts, and logs before making the service available to lab users.
