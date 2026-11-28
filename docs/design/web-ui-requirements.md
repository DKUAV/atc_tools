# ATC Tools Web UI Requirements

The laboratory Web UI takes a user from a local ONNX file to a downloadable OM through a predefined conversion workflow. It offers the targets and presets available in the deployment and shows the execution plan before submission. Users sign in with an account provisioned by a laboratory administrator. [Service requirements](service-requirements.md)

## Task pages

- **My tasks** shows the authenticated user's task history and gives them a clear entry point to start a conversion. Each row shows the model name, target, preset, and current status.
- **New conversion** keeps the model upload, configuration, and execution plan on one page. Upload and inspect the model, select a target and preset, supply the required inputs, and review the plan before pressing **Start**.
- **Task detail** shows stage status, incremental logs, and artifacts. Each step has a status icon and text. Pending, running, and completed stages use a circle, a spinner, and a check mark respectively. Failed and cancelled stages show the outcome and reason.

## Model and preset selection

Upload one local ONNX file with embedded weights. Start inspection after the upload and display the model input names, shapes, and dtypes. Use declared fixed dimensions automatically. If a workflow needs concrete dimensions for dynamic axes, offer an approved shape preset or ask for the missing dimensions. Resolve ambiguous layout or channel order through the input specification.

The **Compile preset** field is one dropdown. Each option selects a complete configuration. The API supplies the compatible choices for the selected target and toolchain.

| Preset | Intended behavior |
| --- | --- |
| Original precision | Use ATC's original precision policy. |
| Float16 automatic mixed precision | Apply the verified ATC mixed precision policy. |
| Float16 preferred | Apply the verified ATC FP16 policy. |
| INT8 default calibration | Use the laboratory's versioned AMCT configuration. |

For 910B2, numerical comparison and `msprof` collection are independent choices. INT8 comparison produces an original ONNX versus INT8 OM report and a same-device floating-point OM versus INT8 OM report. The server defines the steps and their dependencies.

## Datasets and input preparation

Display **Upload calibration set** only for INT8. Display **Upload validation set** when numerical comparison or `msprof` is selected. Each dataset has one entry point offering file selection or folder selection. Users can select multiple files. A selected directory includes its subdirectories. The UI reports the number of recognized files and skipped entries. Accepted dataset formats are JPG, JPEG, PNG, `.npy`, and `.bin`.

Bind samples to named model inputs. Single-input samples can be assigned automatically. Where a sample has multiple tensors, every sample needs a complete mapping. Read the source shape from decoded images or `.npy` metadata where available. Use a user-confirmed or preset source specification for raw `.bin` when its source shape or dtype differs from the model input.

Calibration and validation use **the same input preparation configuration**. Users select both dimensions and numerical treatment whenever the workflow consumes input data. File extensions select the decoder. The chosen configuration determines the transformations.

| Setting | Choices and behavior |
| --- | --- |
| Dimensions | Keep source dimensions, resize to the target, pad to the target, or resize proportionally and pad. Pad alone requires the source to fit inside the target. Both padding choices expose an editable fill value with a default of `0`. |
| Numerical treatment | Use values as supplied, divide by 255, use a standardization preset, or configure standardization. Custom standardization expands per-channel mean and positive standard deviation fields plus a choice to use the supplied values or divide by 255 first. |

Apply dimensions before numerical treatment. Interpret fill values in the source value range. Validate all resulting tensors against the required model input shape and dtype. For a validation run, use the same prepared input for ONNX and the corresponding OM comparisons.

## Submission and delivery

Enable **Start** after model inspection passes and all applicable settings and files are ready. Display the first actionable reason when submission is unavailable. Submit the approved workflow once and navigate to its task details. A user can follow the queue and execution stages, read logs, and download each output as it becomes available. A failed optional stage leaves a successfully produced OM available during the session.

Uploads and downloadable outputs belong to the browser session. Logout and session expiry make them unavailable for download or reuse. The service clears the files when active workflows have finished using them. This covers uploaded inputs, OM files, precision reports, and profiling reports and data. Task status and logs remain tied to the account. On departure, remind users to download anything they need to keep. On return, display that a previous session's files have been cleared.

## Acceptance scenarios

- Submit the available 310B4 compile workflow from a local ONNX file and follow it through inspection, execution, logs, and artifact download.
- Exercise all enabled targets and presets. Check conditional datasets, shared input preparation, input shape validation, and the selected 910B2 verification steps.
- Submit a task, close the page, and verify that processing continues. Check that logout and expiry revoke file access and that final cleanup preserves task history and logs.
- Verify that the UI works with keyboard navigation, indicates failures with text, fits narrow screens, and prevents duplicate submission while the request is pending.
