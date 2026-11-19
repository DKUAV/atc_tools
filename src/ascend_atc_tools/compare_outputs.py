"""Compare two directories of named model outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ascend_atc_tools.tensor_io import Array, parse_pairs, read_outputs


def compare(
    reference: Array, actual: Array, atol: float, rtol: float
) -> tuple[bool, str]:
    """Compare shapes, data types and values using reference-relative tolerance.

    Parameters
    ----------
    reference, actual
        Expected and measured arrays.
    atol, rtol
        Nonnegative absolute and relative tolerance for floating-point arrays.

    Returns
    -------
    tuple[bool, str]
        Whether the tensor passed, followed by the diagnostic text.
    """
    if reference.shape != actual.shape:
        return False, f"shape mismatch: {reference.shape} vs {actual.shape}"
    types = f"{reference.dtype} vs {actual.dtype}"
    if reference.dtype.kind != "f" or actual.dtype.kind != "f":
        if reference.dtype != actual.dtype or reference.dtype.kind not in "biu":
            return False, f"unsupported or incompatible dtypes: {types}"
        count = int(np.count_nonzero(reference != actual))
        return count == 0, f"{types}, unequal={count}/{reference.size}"
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        return False, f"{types}, non-finite values found"
    ref = reference.astype(np.float64)
    got = actual.astype(np.float64)
    with np.errstate(over="ignore", divide="ignore"):
        error = np.abs(got - ref)
        violations = error > atol + rtol * np.abs(ref)
        relative = error / np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
    count = int(np.count_nonzero(violations))
    location = ""
    if count:
        index = np.unravel_index(
            int(np.argmax(np.where(violations, error, -1.0))), ref.shape
        )
        location = f", largest failing error at {tuple(map(int, index))}: ref={ref[index]:.6g}, got={got[index]:.6g}"
    return count == 0, (
        f"{types}, max_abs={float(error.max(initial=0)):.6g}, "
        f"max_rel={float(relative.max(initial=0)):.6g}, violations={count}/{ref.size}{location}"
    )


def main() -> int:
    """Report comparisons and return a failing exit code for any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument(
        "--output-map", action="append", default=[], metavar="REFERENCE=ACTUAL"
    )
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        parser.error("Tolerances must be finite and nonnegative.")
    try:
        reference = read_outputs(args.reference)
        actual = read_outputs(args.actual)
        overrides = parse_pairs(args.output_map)
        if overrides.keys() - reference.keys():
            raise ValueError(
                f"Unknown reference names: {sorted(overrides.keys() - reference.keys())}"
            )
        names = {name: overrides.get(name, name) for name in reference}
        if (
            len(set(names.values())) != len(names)
            or set(names.values()) != actual.keys()
        ):
            raise ValueError(
                f"Output names do not match: expected {sorted(names.values())}, got {sorted(actual)}"
            )
        passed = True
        for name, actual_name in names.items():
            matched, detail = compare(
                reference[name], actual[actual_name], args.atol, args.rtol
            )
            print(f"{'PASS' if matched else 'FAIL'} {name}: {detail}")
            passed &= matched
        return 0 if passed else 1
    except (OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
