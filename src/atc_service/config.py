"""Settings shared by the HTTP service and its worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Store immutable deployment settings for the service."""

    data_dir: Path
    runtime: str
    simulate_compile: bool = False
    max_upload_bytes: int = 512 * 1024 * 1024
    max_log_bytes: int = 4 * 1024 * 1024
    max_output_bytes: int = 1024 * 1024 * 1024
    timeout_seconds: int = 900
    max_pending_jobs: int = 8
    max_account_bytes: int = 4 * 1024 * 1024 * 1024
    browser_secure_cookies: bool = True
    browser_session_seconds: int = 12 * 60 * 60
    amct_targets: tuple[str, ...] = ()
    ascend_device: int | None = None
    cpu_slots: int = 2
    compile_slots: int = 1
    scratch_bytes: int = 4 * 1024**3

    @classmethod
    def from_env(cls) -> Settings:
        """Read configuration supplied by the deployment.

        Returns
        -------
        Settings
            Service, storage, and task limits.
        """
        return cls(
            data_dir=Path(os.environ.get("ATC_DATA_DIR", "/data")),
            runtime=runtime_identity(),
            simulate_compile=os.environ.get("ATC_SIMULATE_COMPILE") == "1",
            max_upload_bytes=int(
                os.environ.get("ATC_MAX_UPLOAD_BYTES", 512 * 1024 * 1024)
            ),
            max_log_bytes=int(os.environ.get("ATC_MAX_LOG_BYTES", 4 * 1024 * 1024)),
            max_output_bytes=int(
                os.environ.get("ATC_MAX_OUTPUT_BYTES", 1024 * 1024 * 1024)
            ),
            timeout_seconds=int(os.environ.get("ATC_TIMEOUT_SECONDS", 900)),
            max_pending_jobs=int(os.environ.get("ATC_MAX_PENDING_JOBS", 8)),
            max_account_bytes=int(
                os.environ.get("ATC_MAX_ACCOUNT_BYTES", 4 * 1024 * 1024 * 1024)
            ),
            browser_secure_cookies=os.environ.get("ATC_BROWSER_SECURE_COOKIES") != "0",
            browser_session_seconds=int(
                os.environ.get("ATC_BROWSER_SESSION_SECONDS", 43200)
            ),
            amct_targets=tuple(
                filter(None, os.environ.get("ATC_AMCT_TARGETS", "").split(","))
            ),
            ascend_device=int(device)
            if (device := os.environ.get("ATC_910B2_DEVICE"))
            else None,
            cpu_slots=int(os.environ.get("ATC_CPU_SLOTS", 2)),
            compile_slots=int(os.environ.get("ATC_COMPILE_SLOTS", 1)),
            scratch_bytes=int(os.environ.get("ATC_SCRATCH_BYTES", 4 * 1024**3)),
        )

    def __post_init__(self) -> None:
        """Validate execution limits and configured target devices.

        Raises
        ------
        ValueError
            A resource limit or device configuration is invalid.
        """
        if not set(self.amct_targets) <= {"310B4", "910B2"}:
            raise ValueError("ATC_AMCT_TARGETS may contain 310B4 and 910B2.")
        if self.ascend_device is not None and self.ascend_device < 0:
            raise ValueError("ATC_910B2_DEVICE must be nonnegative.")
        if (
            min(
                self.max_upload_bytes,
                self.max_log_bytes,
                self.max_output_bytes,
                self.timeout_seconds,
                self.max_pending_jobs,
                self.max_account_bytes,
                self.browser_session_seconds,
                self.cpu_slots,
                self.compile_slots,
                self.scratch_bytes,
            )
            <= 0
        ):
            raise ValueError("The service resource limits must be positive.")

    def compile_preset(self) -> dict[str, str | bool]:
        """Return the enabled 310B4 preset.

        Returns
        -------
        dict[str, str | bool]
            The public preset for the installed compiler or the simulator.
        """
        if self.simulate_compile:
            return {"id": "310b4-simulation", "target": "310B4", "simulated": True}
        return {"id": "310b4-float", "target": "310B4", "simulated": False}


def runtime_identity() -> str:
    """Identify the currently installed tools and the base toolchain.

    Returns
    -------
    str
        Build identity including local wheel updates.
    """
    return (
        f"{os.environ.get('ATC_BASE_IMAGE', 'local')}/atc_tools-{version('atc_tools')}"
    )
