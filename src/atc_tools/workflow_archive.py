"""Pack and read bounded per-sample tensor results for standalone operations."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def pack_results(root: Path, destination: Path, simulated: bool, limit: int) -> None:
    """Package the prepared sample outputs and their execution mode.

    Parameters
    ----------
    root, destination
        Per-sample tensor directory and output archive.
    simulated
        Whether the outputs substitute for NPU inference.
    limit
        Maximum total bytes of uncompressed tensors and sample manifests.
    """
    files = sorted(root.glob("*/*"))
    if not files or sum(path.stat().st_size for path in files) > limit:
        raise ValueError("Tensor outputs are empty or exceed the output budget.")
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps({"simulated": simulated}))
        for path in files:
            archive.write(path, path.relative_to(root))


def unpack_results(source: Path, root: Path, limit: int) -> bool:
    """Unpack only the generated tensor paths within a bounded private directory.

    Parameters
    ----------
    source, root
        Owned tensor archive and private extraction destination.
    limit
        Maximum total uncompressed bytes of tensors and sample manifests.
        The top-level archive manifest has its own size limit.

    Returns
    -------
    bool
        Whether the archive was produced by simulated inference.
    """
    with ZipFile(source) as archive:
        members = archive.infolist()
        names = {entry.filename for entry in members}
        payload_size = sum(
            entry.file_size for entry in members if entry.filename != "manifest.json"
        )
        if (
            len(names) != len(members)
            or payload_size > limit
            or "manifest.json" not in names
        ):
            raise ValueError("Invalid or oversized tensor archive.")
        manifest = archive.getinfo("manifest.json")
        if manifest.file_size > 256:
            raise ValueError("Invalid tensor archive manifest.")
        simulated = json.loads(archive.read(manifest)).get("simulated")
        if type(simulated) is not bool:
            raise ValueError("Unknown tensor archive execution mode.")
        for entry in members:
            if entry.filename == "manifest.json":
                continue
            if not re.fullmatch(
                r"[0-9]{8}/(?:outputs\.json|output_[0-9]+\.npy)", entry.filename
            ):
                raise ValueError("Unexpected tensor archive member.")
            target = root / entry.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
        if not root.is_dir():
            raise ValueError("The tensor archive is empty.")
        return simulated
