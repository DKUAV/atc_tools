"""Define fixed conversion presets and validate browser workflow inputs."""

from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from atc_service.config import Settings
from atc_service.store import ResourceError

DATASET_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".npy", ".bin"})
PositiveDimension = Annotated[int, Field(strict=True, ge=1, le=65536)]
Shape = Annotated[list[PositiveDimension], Field(max_length=16)]
Sample = Annotated[list[UUID], Field(min_length=1, max_length=16)]


class Selection(BaseModel):
    """Reject unknown fields in browser workflow selections."""

    model_config = ConfigDict(extra="forbid")


class RawTensor(Selection):
    """Optionally describe the raw .bin format for one ONNX input."""

    shape: Shape
    dtype: Literal[
        "float32",
        "float16",
        "float64",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    ]


class Preparation(Selection):
    """Describe the shared geometry and numerical transformation."""

    geometry: Literal["keep", "resize", "pad", "letterbox"] = "keep"
    layout: Literal["auto", "nchw", "nhwc"] = "auto"
    fill: FiniteFloat = 0
    values: Literal["raw", "unit", "imagenet", "custom"] = "raw"
    mean: list[FiniteFloat] = Field(default_factory=list, max_length=16)
    std: list[Annotated[FiniteFloat, Field(gt=0)]] = Field(
        default_factory=list, max_length=16
    )
    unit_first: bool = False
    raw_inputs: dict[Annotated[int, Field(ge=0, lt=16)], RawTensor] = Field(
        default_factory=dict, max_length=16
    )


class Conversion(Selection):
    """Select the model, preset, and inputs for one fixed workflow."""

    model_id: UUID
    preset: str = Field(min_length=1, max_length=64)
    compare: bool = False
    profile: bool = False
    calibration: list[Sample] = Field(default_factory=list, max_length=128)
    validation: list[Sample] = Field(default_factory=list, max_length=128)
    shapes: dict[Annotated[int, Field(ge=0, lt=16)], Shape] = Field(
        default_factory=dict, max_length=16
    )
    preparation: Preparation = Field(default_factory=Preparation)


class Operation(Selection):
    """Select a standalone workflow operation and its owned input resources."""

    operation: Literal[
        "quantize_onnx",
        "compile_om",
        "run_cpu",
        "run_npu",
        "compare_outputs",
        "profile_om",
    ]
    input_id: UUID
    preset: str | None = Field(default=None, max_length=64)
    primary_id: UUID | None = None
    actual_id: UUID | None = None
    samples: list[Sample] = Field(default_factory=list, max_length=128)
    shapes: dict[Annotated[int, Field(ge=0, lt=16)], Shape] = Field(
        default_factory=dict, max_length=16
    )
    preparation: Preparation = Field(default_factory=Preparation)


class StepRetry(Selection):
    """Select a failed optional stage and, optionally, replacement validation data."""

    step: Literal["compare", "profile"]
    validation: list[Sample] | None = Field(default=None, min_length=1, max_length=128)


def resolve_operation(
    body: Operation, presets: dict[str, "Preset"], settings: Settings
) -> tuple[str, bool, dict[str, object]]:
    """Resolve an atomic operation to a trusted image and a fixed task plan.

    Parameters
    ----------
    body
        Validated operation request.
    presets
        Enabled workflow presets by public ID.
    settings
        Deployment image and simulator configuration.

    Returns
    -------
    tuple
        Executor image, simulation flag, and immutable job spec.

    Raises
    ------
    ResourceError
        A target, resource or setting is missing or inapplicable.
    """
    action = body.operation
    target = presets.get(body.preset or "")
    if (
        action in {"compile_om", "quantize_onnx", "run_npu", "profile_om"}
        and target is None
    ):
        raise ResourceError(422, "Select an enabled preset for the operation.")
    if action in {"run_cpu", "compare_outputs"} and body.preset is not None:
        raise ResourceError(
            422, "This operation uses the CPU executor without a target preset."
        )
    if action == "quantize_onnx" and (target is None or target.precision != "int8"):
        raise ResourceError(422, "Select an enabled INT8 preset for quantization.")
    npu = action in {"run_npu", "profile_om"}
    if npu and (target is None or not target.checks or target.target != "910B2"):
        raise ResourceError(422, "Select an enabled 910B2 runtime preset.")
    if (body.primary_id is not None) != npu or (body.actual_id is not None) != (
        action == "compare_outputs"
    ):
        raise ResourceError(
            422,
            "Provide the OM or second tensor archive only for the matching operation.",
        )
    if bool(body.samples) != (
        action in {"quantize_onnx", "run_cpu", "run_npu", "profile_om"}
    ):
        raise ResourceError(
            422, "Provide samples for quantization or inference and profiling."
        )
    if (
        body.shapes
        and action == "compare_outputs"
        or body.preparation != Preparation()
        and not body.samples
    ):
        raise ResourceError(
            422, "This operation does not accept those input preparation settings."
        )
    if body.preparation.values == "custom" and (
        not body.preparation.mean
        or len(body.preparation.mean) != len(body.preparation.std)
    ):
        raise ResourceError(
            422, "Enter matching mean and positive standard deviation values."
        )
    if len({len(sample) for sample in body.samples}) > 1:
        raise ResourceError(422, "Every sample needs the same number of model inputs.")
    data = body.model_dump(mode="json")
    return (
        target.image if target else settings.inspect_image,
        bool(target and target.simulated),
        {
            "action": action,
            "target": target.target if target else None,
            "precision": target.precision if target else None,
            "calibration": data["samples"] if action == "quantize_onnx" else [],
            "validation": data["samples"]
            if action in {"run_cpu", "run_npu", "profile_om"}
            else [],
            "compare": False,
            "profile": False,
            "shapes": data["shapes"],
            "preparation": data["preparation"],
            "primary_id": data["primary_id"],
            "actual_id": data["actual_id"],
        },
    )


@dataclass(frozen=True)
class Preset:
    """Bind one visible preset to a trusted executor and fixed behavior."""

    id: str
    target: str
    precision: str
    image: str
    simulated: bool
    checks: bool

    def public(self) -> dict[str, str | bool]:
        """Return the fields safe to expose to the browser.

        Returns
        -------
        dict
            Preset selection and optional-stage capability.
        """
        return {
            "id": self.id,
            "target": self.target,
            "precision": self.precision,
            "simulated": self.simulated,
            "checks": self.checks,
        }


def browser_presets(settings: Settings) -> list[Preset]:
    """List the enabled browser workflows in target and precision order.

    Parameters
    ----------
    settings
        Deployment's trusted executors and device configuration.

    Returns
    -------
    list[Preset]
        Selectable compile presets.
    """
    presets = []
    for target, configured in (
        ("310B4", settings.compile_image),
        ("910B2", settings.compile_910b2_image),
    ):
        image = settings.inspect_image if settings.simulate_compile else configured
        if image is None:
            continue
        checks = target == "910B2" and (
            settings.simulate_compile or settings.ascend_device is not None
        )
        for precision in ("origin", "mixed_float16", "fp16", "int8"):
            if (
                precision == "int8"
                and not settings.simulate_compile
                and target not in settings.amct_targets
            ):
                continue
            presets.append(
                Preset(
                    f"{target.lower()}-{precision}",
                    target,
                    precision,
                    image,
                    settings.simulate_compile,
                    checks,
                )
            )
    return presets


def resolve_workflow(body: Conversion, preset: Preset) -> dict[str, object]:
    """Construct the immutable executor request for a valid workflow.

    Parameters
    ----------
    body
        Validated model, options, and sample selection.
    preset
        Workflow configuration derived from deployment settings.

    Returns
    -------
    dict
        Resolved workflow and dataset asset IDs in sample/input order.

    Raises
    ------
    ResourceError
        A required dataset or option is missing or incompatible.
    """
    if (body.compare or body.profile) and not preset.checks:
        raise ResourceError(
            422, "Comparison and profiling require an enabled 910B2 runtime."
        )
    if bool(body.calibration) != (preset.precision == "int8"):
        raise ResourceError(422, "Provide a calibration set for INT8 only.")
    if bool(body.validation) != (body.compare or body.profile):
        raise ResourceError(
            422, "Provide a validation set when comparison or profiling is selected."
        )
    prep = body.preparation
    if prep.values == "custom" and (not prep.mean or len(prep.mean) != len(prep.std)):
        raise ResourceError(
            422, "Enter matching mean and positive standard deviation values."
        )
    if len({len(sample) for sample in [*body.calibration, *body.validation]}) > 1:
        raise ResourceError(422, "Every sample needs the same number of model inputs.")
    return {
        "target": preset.target,
        "precision": preset.precision,
        **body.model_dump(mode="json", exclude={"model_id", "preset"}),
    }


def planned_stages(workflow: dict[str, object]) -> list[str]:
    """Return the fixed stage order for an accepted workflow.

    Parameters
    ----------
    workflow
        Resolved workflow settings with fixed dependencies.

    Returns
    -------
    list[str]
        Stage IDs in execution order.
    """
    action = workflow.get("action")
    if action == "retry_step":
        stages = ["compile_baseline"] if workflow.get("build_baseline") else []
        return [*stages, "prepare_validation", str(workflow["retry_step"])]
    atomic = {
        "quantize_onnx": ["prepare_calibration", "quantize"],
        "compile_om": ["compile"],
        "run_cpu": ["prepare_validation", "infer"],
        "run_npu": ["prepare_validation", "infer"],
        "compare_outputs": ["compare"],
        "profile_om": ["prepare_validation", "profile"],
    }
    if isinstance(action, str):
        return atomic[action]
    stages = []
    if workflow["precision"] == "int8":
        stages.extend(("prepare_calibration", "quantize"))
    stages.append("compile")
    if workflow["precision"] == "int8" and workflow["compare"]:
        stages.append("compile_baseline")
    if workflow["compare"] or workflow["profile"]:
        stages.append("prepare_validation")
    if workflow["compare"]:
        stages.append("compare")
    if workflow["profile"]:
        stages.append("profile")
    return stages
