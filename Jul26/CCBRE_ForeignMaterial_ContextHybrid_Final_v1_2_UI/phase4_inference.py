#!/usr/bin/env python3
"""Final three-stage T-Core+Other inference used by testing and the visual app.

v1.2 fix:
- detector boxes are sanitized/clipped before crop extraction;
- degenerate/out-of-frame proposals are skipped safely;
- rendering also clips boxes before drawing;
- prevents Pillow: "Coordinate 'lower' is less than 'upper'".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


def norm(value):
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def font(size):
    for path in (
        r"C:\Windows\Fonts\arial.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sanitize_box(box, size, minimum_side=1.0):
    """Return a clipped valid xyxy box, or None for a degenerate proposal."""
    width, height = size
    if width <= 0 or height <= 0:
        return None
    try:
        x1, y1, x2, y2 = map(float, box)
    except Exception:
        return None

    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    # Be defensive about any accidental reversed xyxy coordinates.
    left, right = sorted((x1, x2))
    upper, lower = sorted((y1, y2))

    left = max(0.0, min(float(width), left))
    right = max(0.0, min(float(width), right))
    upper = max(0.0, min(float(height), upper))
    lower = max(0.0, min(float(height), lower))

    if right - left < minimum_side or lower - upper < minimum_side:
        return None
    return [left, upper, right, lower]


def expand(box, size, margin, minimum_side=2):
    """Expand and clip xyxy; return a valid Pillow crop tuple or None."""
    clean = sanitize_box(box, size, minimum_side=1.0)
    if clean is None:
        return None

    width, height = size
    x1, y1, x2, y2 = clean
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    left = max(0, min(width, int(math.floor(x1 - margin * bw))))
    upper = max(0, min(height, int(math.floor(y1 - margin * bh))))
    right = max(0, min(width, int(math.ceil(x2 + margin * bw))))
    lower = max(0, min(height, int(math.ceil(y2 + margin * bh))))

    # Pillow requires right >= left and lower >= upper. For inference,
    # a zero-/one-pixel crop is not useful, so reject it.
    if right - left < minimum_side or lower - upper < minimum_side:
        return None
    return (left, upper, right, lower)


@dataclass
class Detection:
    bbox: list[float]
    detector_confidence: float
    verifier_fo_probability: float
    accepted_by: str
    class_id: int | None
    class_name: str | None
    exact_confidence: float
    exact_margin: float
    label: str


class Pipeline:
    def __init__(self, detector, verifier, exact, settings, device="0"):
        from ultralytics import YOLO

        self.detector_path = Path(detector)
        self.verifier_path = Path(verifier)
        self.exact_path = Path(exact)
        self.s = settings
        self.device = str(device)

        for name, path in (
            ("detector", self.detector_path),
            ("verifier", self.verifier_path),
            ("exact classifier", self.exact_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{name} checkpoint not found: {path}"
                )

        self.detector = YOLO(str(self.detector_path), task="detect")
        self.verifier = YOLO(str(self.verifier_path), task="classify")
        self.exact = YOLO(str(self.exact_path), task="classify")

        verifier_names = {
            int(key): str(value)
            for key, value in self.verifier.names.items()
        }
        self.exact_names = {
            int(key): str(value)
            for key, value in self.exact.names.items()
        }
        by_name = {
            norm(value): key
            for key, value in verifier_names.items()
        }
        self.fo = by_name.get(norm("ForeignMaterial"))
        self.bg = by_name.get(norm("Background_NotFO"))
        if self.fo is None or self.bg is None:
            raise RuntimeError(
                f"Binary verifier classes invalid: {verifier_names}"
            )

    def analyze(self, image_path):
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        settings = self.s
        detector_result = self.detector.predict(
            source=str(image_path),
            imgsz=int(settings["detector_imgsz"]),
            conf=float(settings["detector_inference_floor"]),
            iou=float(settings["nms_iou"]),
            max_det=int(settings["max_detections"]),
            device=self.device,
            verbose=False,
        )[0]

        proposals = []
        invalid_detector_proposals = 0
        if (
            detector_result.boxes is not None
            and len(detector_result.boxes)
        ):
            boxes = (
                detector_result.boxes.xyxy
                .detach().float().cpu().numpy()
            )
            confidences = (
                detector_result.boxes.conf
                .detach().float().cpu().numpy()
            )
            for box, confidence in zip(boxes, confidences):
                clean = sanitize_box(box, image.size)
                if clean is None:
                    invalid_detector_proposals += 1
                    continue
                proposals.append(
                    {
                        "bbox": clean,
                        "detector_confidence": float(confidence),
                    }
                )

        empty_result = {
            "image": str(image_path.resolve()),
            "foreign_material_present": False,
            "detections": [],
            "diagnostics": {
                "raw_detector_proposals": (
                    len(proposals) + invalid_detector_proposals
                ),
                "valid_detector_proposals": len(proposals),
                "invalid_detector_proposals_skipped": (
                    invalid_detector_proposals
                ),
                "invalid_verifier_crops_skipped": 0,
                "invalid_exact_crops_skipped": 0,
            },
        }
        if not proposals:
            return image, empty_result

        verifier_items = []
        verifier_crops = []
        invalid_verifier_crops = 0
        for proposal in proposals:
            crop_box = expand(
                proposal["bbox"],
                image.size,
                float(settings["verifier_crop_margin"]),
            )
            if crop_box is None:
                invalid_verifier_crops += 1
                continue
            verifier_items.append(proposal)
            verifier_crops.append(image.crop(crop_box))

        if not verifier_items:
            empty_result["diagnostics"][
                "invalid_verifier_crops_skipped"
            ] = invalid_verifier_crops
            return image, empty_result

        verifier_results = self.verifier.predict(
            source=verifier_crops,
            imgsz=int(settings["verifier_imgsz"]),
            batch=min(24, len(verifier_crops)),
            device=self.device,
            verbose=False,
        )

        accepted = []
        for proposal, result in zip(
            verifier_items,
            verifier_results,
        ):
            probabilities = (
                result.probs.data.detach().float().cpu().numpy()
            )
            fo_probability = float(probabilities[self.fo])
            detector_confidence = proposal["detector_confidence"]

            normal = (
                detector_confidence
                >= float(settings["detector_confidence"])
                and fo_probability
                >= float(settings["verifier_fo_threshold"])
            )
            rescue = (
                bool(settings.get("rescue_enabled", False))
                and detector_confidence
                >= float(settings["rescue_detector_confidence"])
                and fo_probability
                >= float(settings["rescue_verifier_fo_threshold"])
            )
            if normal or rescue:
                accepted.append(
                    {
                        **proposal,
                        "verifier_fo_probability": fo_probability,
                        "accepted_by": (
                            "normal" if normal else "rescue"
                        ),
                    }
                )

        if not accepted:
            empty_result["diagnostics"][
                "invalid_verifier_crops_skipped"
            ] = invalid_verifier_crops
            return image, empty_result

        exact_items = []
        exact_crops = []
        invalid_exact_crops = 0
        for proposal in accepted:
            crop_box = expand(
                proposal["bbox"],
                image.size,
                float(settings["exact_crop_margin"]),
            )
            if crop_box is None:
                invalid_exact_crops += 1
                continue
            exact_items.append(proposal)
            exact_crops.append(image.crop(crop_box))

        if not exact_items:
            empty_result["diagnostics"][
                "invalid_verifier_crops_skipped"
            ] = invalid_verifier_crops
            empty_result["diagnostics"][
                "invalid_exact_crops_skipped"
            ] = invalid_exact_crops
            return image, empty_result

        exact_results = self.exact.predict(
            source=exact_crops,
            imgsz=int(settings["exact_imgsz"]),
            batch=min(32, len(exact_crops)),
            device=self.device,
            verbose=False,
        )

        detections = []
        for proposal, result in zip(exact_items, exact_results):
            probabilities = (
                result.probs.data.detach().float().cpu().numpy()
            )
            order = probabilities.argsort()[::-1]
            top1 = int(order[0])
            top2 = int(order[1]) if len(order) > 1 else top1
            confidence = float(probabilities[top1])
            margin = confidence - float(probabilities[top2])
            exact_ok = (
                confidence >= float(settings["exact_confidence"])
                and margin >= float(settings["exact_margin"])
            )
            class_name = (
                self.exact_names[top1] if exact_ok else None
            )
            class_id = top1 if exact_ok else None
            label = (
                class_name
                if class_name
                else "ForeignMaterial (class uncertain)"
            )
            label += (
                f' | Det={proposal["detector_confidence"]:.2f}'
                f' Ver={proposal["verifier_fo_probability"]:.2f}'
                f" Cls={confidence:.2f}"
            )
            detections.append(
                Detection(
                    proposal["bbox"],
                    proposal["detector_confidence"],
                    proposal["verifier_fo_probability"],
                    proposal["accepted_by"],
                    class_id,
                    class_name,
                    confidence,
                    margin,
                    label,
                )
            )

        return image, {
            "image": str(image_path.resolve()),
            "foreign_material_present": bool(detections),
            "detections": [asdict(item) for item in detections],
            "diagnostics": {
                "raw_detector_proposals": (
                    len(proposals) + invalid_detector_proposals
                ),
                "valid_detector_proposals": len(proposals),
                "invalid_detector_proposals_skipped": (
                    invalid_detector_proposals
                ),
                "invalid_verifier_crops_skipped": (
                    invalid_verifier_crops
                ),
                "invalid_exact_crops_skipped": invalid_exact_crops,
            },
        }

    @staticmethod
    def render(image, result):
        output = image.copy()
        draw = ImageDraw.Draw(output)
        selected_font = font(
            max(15, round(min(output.size) / 50))
        )

        if not result["detections"]:
            message = "NO FOREIGN MATERIAL DETECTED"
            text_box = draw.textbbox(
                (0, 0),
                message,
                font=selected_font,
            )
            draw.rectangle(
                (
                    10,
                    10,
                    text_box[2] - text_box[0] + 30,
                    text_box[3] - text_box[1] + 30,
                ),
                fill=(20, 120, 40),
            )
            draw.text(
                (20, 20),
                message,
                font=selected_font,
                fill="white",
            )
            return output

        line_width = max(
            3,
            round(min(output.size) / 400),
        )
        for detection in result["detections"]:
            clean = sanitize_box(
                detection["bbox"],
                output.size,
            )
            if clean is None:
                continue
            x1, y1, x2, y2 = clean
            draw.rectangle(
                (x1, y1, x2, y2),
                outline=(225, 30, 30),
                width=line_width,
            )
            message = detection["label"]
            text_box = draw.textbbox(
                (0, 0),
                message,
                font=selected_font,
            )
            text_height = text_box[3] - text_box[1]
            label_y = max(0.0, y1 - text_height - 8)
            label_right = min(
                float(output.width),
                x1 + text_box[2] - text_box[0] + 8,
            )
            label_lower = min(
                float(output.height),
                label_y + text_height + 6,
            )
            if label_right > x1 and label_lower > label_y:
                draw.rectangle(
                    (x1, label_y, label_right, label_lower),
                    fill=(225, 30, 30),
                )
            draw.text(
                (x1 + 4, label_y + 2),
                message,
                font=selected_font,
                fill="white",
            )
        return output
