"""Run YOLO on one image with ais_bench or prepare and render tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from atc_tools.inference.ais_bench import create_session, describe_inputs
from atc_tools.model.image_input import image_tensor, letterbox, read_image
from atc_tools.model.tensor_io import image_geometry


def infer_image(
    image: NDArray[np.uint8], model: Path, device: int
) -> tuple[NDArray[np.generic], float, int, int]:
    """Run one YOLO image through a static OM and preserve its letterbox transform.

    Parameters
    ----------
    image
        Original BGR image.
    model
        OM with one image input and one prediction output.
    device
        Logical NPU device ID.

    Returns
    -------
    tuple[NDArray[np.generic], float, int, int]
        Prediction tensor, resize scale, and left and top padding.

    Raises
    ------
    ValueError
        The OM does not have the expected image input and prediction output.
    RuntimeError
        The ais_bench backend failed or did not return a prediction tensor.
    """
    session = create_session(model, device)
    try:
        inputs = describe_inputs(session)
        if len(inputs) != 1 or len(session.get_outputs()) != 1:
            raise ValueError(
                "YOLO inference requires one input and one prediction output."
            )
        spec = inputs[0]
        channels_first, height, width = image_geometry(spec)
        canvas, scale, left, top = letterbox(image, width, height)
        tensor = image_tensor(canvas, channels_first, spec.dtype)
        predictions: list[NDArray[np.generic]] = session.infer([tensor], mode="static")
        if len(predictions) != 1 or not isinstance(predictions[0], np.ndarray):
            raise RuntimeError(
                "ais_bench did not return a single YOLO prediction tensor."
            )
        return predictions[0], scale, left, top
    finally:
        session.free_resource()


def detections(
    prediction: NDArray[np.generic], layout: str, confidence: float, iou: float
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    """Decode batch-one YOLO predictions and apply class-aware NMS if needed.

    Parameters
    ----------
    prediction
        Raw model output for a single image.
    layout
        Detection output layout: v5, v8, or xyxy.
    confidence, iou
        Confidence filter and IoU threshold for NMS.

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]
        Boxes in model input coordinates, scores, and class IDs.

    Raises
    ------
    ValueError
        The tensor does not match the chosen layout or contains non-finite values.
    """
    if prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.dtype.kind != "f":
        raise ValueError(
            f"Expected one floating-point prediction batch, got {prediction.shape}"
        )
    rows = prediction[0].astype(np.float32)
    if not np.isfinite(rows).all():
        raise ValueError("Predictions contain non-finite values.")
    if layout == "xyxy":
        if rows.shape[1] != 6:
            raise ValueError("The xyxy layout requires [1, N, 6].")
        selected = rows[:, 4] >= confidence
        return rows[selected, :4], rows[selected, 4], rows[selected, 5].astype(np.int64)
    if layout == "v8":
        rows = rows.T
    class_start = 5 if layout == "v5" else 4
    if rows.shape[1] <= class_start:
        raise ValueError(f"The {layout} layout has no class scores: {prediction.shape}")
    classes = rows[:, class_start:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), class_start + classes]
    if layout == "v5":
        scores = scores * rows[:, 4]
    selected = (scores >= confidence) & (rows[:, 2:4] > 0).all(axis=1)
    xywh, scores, classes = (
        rows[selected, :4].copy(),
        scores[selected],
        classes[selected],
    )
    xywh[:, :2] -= xywh[:, 2:4] / 2
    keep: list[int] = []
    for label in np.unique(classes):
        indices = np.flatnonzero(classes == label)
        local = cv2.dnn.NMSBoxes(
            xywh[indices].tolist(), scores[indices].tolist(), 0.0, iou
        )
        keep.extend(indices[np.asarray(local, dtype=np.intp).ravel()].tolist())
    kept = np.asarray(sorted(keep, key=lambda index: -scores[index]), dtype=np.intp)
    boxes = xywh[kept]
    boxes[:, 2:] += boxes[:, :2]
    return boxes, scores[kept], classes[kept].astype(np.int64)


def run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Run YOLO inference or an individual image processing step.

    Parameters
    ----------
    args
        Parsed options for the selected YOLO command.
    parser
        Parser used to report invalid options or detection errors.

    Raises
    ------
    SystemExit
        The options are invalid or the selected operation failed.
    """
    if args.command == "infer":
        if args.device < 0:
            parser.error("Device ID must be nonnegative.")
        if not args.model.is_file():
            parser.error(f"OM file not found: {args.model}")
        if args.output_dir.exists():
            parser.error(f"Choose a new output directory: {args.output_dir}")
    elif min(args.width, args.height) <= 0:
        parser.error("Image dimensions must be positive.")
    try:
        if args.command != "preprocess" and (
            not 0 < args.confidence <= 1 or not 0 <= args.iou <= 1
        ):
            raise ValueError("Confidence must be in (0, 1] and IoU must be in [0, 1].")
        image = read_image(args.image)
        if args.command == "infer":
            prediction, scale, left, top = infer_image(image, args.model, args.device)
        else:
            canvas, scale, left, top = letterbox(image, args.width, args.height)
            if args.command == "preprocess":
                if args.output.suffix.lower() != ".npy":
                    raise ValueError("The output path must end in .npy.")
                tensor = image_tensor(
                    canvas, channels_first=True, dtype=np.dtype(args.dtype)
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("xb") as stream:
                    np.save(stream, tensor, allow_pickle=False)
                return
            prediction = np.load(args.predictions, allow_pickle=False)
            if not isinstance(prediction, np.ndarray):
                prediction.close()
                raise ValueError("Expected a single .npy prediction tensor.")
        boxes, scores, classes = detections(
            prediction, args.layout, args.confidence, args.iou
        )
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / scale).clip(0, image.shape[1])
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / scale).clip(0, image.shape[0])
        output = []
        for box, score, label in zip(boxes, scores, classes, strict=True):
            x1, y1, x2, y2 = map(float, np.asarray(box).tolist())
            if x2 <= x1 or y2 <= y1:
                continue
            output.append(
                {
                    "xyxy": [x1, y1, x2, y2],
                    "score": float(score),
                    "class_id": int(label),
                }
            )
            start, end = (round(x1), round(y1)), (round(x2), round(y2))
            cv2.rectangle(image, start, end, (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{label}: {score:.2f}",
                start,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        encoded, preview = cv2.imencode(".jpg", image)
        if not encoded:
            raise ValueError("Could not encode annotated image.")
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "detections.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        preview.tofile(args.output_dir / "annotated.jpg")
    except (OSError, RuntimeError, TypeError, ValueError, cv2.error) as error:
        parser.exit(1, f"YOLO example failed: {error}\n")
